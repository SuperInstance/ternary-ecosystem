//! # Ternary Ecosystem: Emergent Behavior from Local Z₃ Interactions
//!
//! ## The Core Claim
//!
//! In a grid of agents competing under Z₃ rock-paper-scissors rules:
//!
//!   Producers (+1) beat Decomposers (0)
//!   Decomposers (0) beat Consumers (-1)
//!   Consumers (-1) beat Producers (+1)
//!
//! **No single species wins. All three coexist indefinitely — but only when
//! interactions are LOCAL.** Switch to mean-field (all-to-all) and one species
//! wins. Coexistence is not programmed; it emerges from the topology.
//!
//! ## Four Mechanisms
//!
//! 1. **Z₃ competition**: local RPS dynamics with zero-sum conservation.
//!    Every death is a birth. Total population is invariant.
//!
//! 2. **Pheromone communication**: species emit charge-weighted signals that
//!    diffuse and decay. Positive pheromone = producer territory. Negative =
//!    consumer territory. Clusters self-reinforce without explicit "go here."
//!
//! 3. **Warp consensus**: groups of WARP_SIZE cells vote. A supermajority
//!    (≥3/4) triggers an amplification event, converting one opposing agent.
//!    Simulates GPU `__ballot_sync` + `__reduce_add_sync`. Creates
//!    super-organism behavior: the warp acts as one decision unit.
//!
//! 4. **CRDT population tracking**: PN-counters track births and deaths per
//!    node. Merge is commutative, associative, idempotent. The conservation
//!    invariant (total_net = 0) holds under any sequence of merges.
//!
//! ## What Tests Prove
//!
//! - Spatial mode: all three species survive 400 steps → coexistence emerged
//! - Mean-field mode: diversity collapses → proves topology created coexistence
//! - Spatial autocorrelation rises over time → clusters self-organized
//! - Total population never changes → conservation law is exact
//! - CRDT algebraic laws hold independently of simulation state

use std::collections::HashMap;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Agents per cell — invariant under all competition events.
pub const CAPACITY: u32 = 12;
/// Cells per simulated GPU warp (real CUDA uses 32; 4 keeps tests fast).
pub const WARP_SIZE: usize = 4;
const PHEROMONE_DECAY: f32 = 0.82;
const PHEROMONE_EMISSION: i32 = 2;

// ─── Deterministic PRNG (xorshift64, no dependencies) ─────────────────────────

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
    fn next_usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }
    fn next_u32_max(&mut self, n: u32) -> u32 { (self.next() as u32) % n }
}

// ─── Z₃ species type ──────────────────────────────────────────────────────────

/// A species in Z₃: its trit value determines its role in the ecosystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Producer,   // trit = +1
    Consumer,   // trit = -1
    Decomposer, // trit =  0
}

impl Species {
    /// The ternary vote this species casts.
    pub fn trit(self) -> i32 {
        match self { Self::Producer => 1, Self::Consumer => -1, Self::Decomposer => 0 }
    }

    /// The species this one outcompetes. Z₃ cycle: P→D→C→P.
    pub fn beats(self) -> Self {
        match self {
            Self::Producer   => Self::Decomposer,
            Self::Decomposer => Self::Consumer,
            Self::Consumer   => Self::Producer,
        }
    }

    /// The species that outcompetes this one.
    pub fn beaten_by(self) -> Self {
        match self {
            Self::Producer   => Self::Consumer,
            Self::Consumer   => Self::Decomposer,
            Self::Decomposer => Self::Producer,
        }
    }

    fn birth_idx(self) -> usize {
        match self { Self::Producer => 0, Self::Consumer => 1, Self::Decomposer => 2 }
    }
    fn death_idx(self) -> usize {
        match self { Self::Producer => 3, Self::Consumer => 4, Self::Decomposer => 5 }
    }
}

// ─── Cell ─────────────────────────────────────────────────────────────────────

/// One grid cell. Holds CAPACITY agents split across three species.
///
/// Invariant: `producers + consumers + decomposers == CAPACITY` at all times.
/// Pheromone is a separate signed signal; it doesn't count toward CAPACITY.
#[derive(Clone, Debug, Default)]
pub struct Cell {
    pub producers: u32,
    pub consumers: u32,
    pub decomposers: u32,
    pub pheromone: i32,
}

impl Cell {
    pub fn new(p: u32, c: u32, d: u32) -> Self {
        debug_assert_eq!(p + c + d, CAPACITY, "cell must sum to CAPACITY");
        Cell { producers: p, consumers: c, decomposers: d, pheromone: 0 }
    }

    pub fn total(&self) -> u32 {
        self.producers + self.consumers + self.decomposers
    }

    pub fn get(&self, s: Species) -> u32 {
        match s {
            Species::Producer   => self.producers,
            Species::Consumer   => self.consumers,
            Species::Decomposer => self.decomposers,
        }
    }

    fn add(&mut self, s: Species, delta: i32) {
        let v = match s {
            Species::Producer   => &mut self.producers,
            Species::Consumer   => &mut self.consumers,
            Species::Decomposer => &mut self.decomposers,
        };
        *v = (*v as i32 + delta).max(0) as u32;
    }

    /// The plurality species (ties broken: Producer > Decomposer > Consumer).
    pub fn dominant(&self) -> Species {
        if self.producers >= self.consumers && self.producers >= self.decomposers {
            Species::Producer
        } else if self.decomposers >= self.consumers {
            Species::Decomposer
        } else {
            Species::Consumer
        }
    }

    /// Ternary vote: sign of (producers − consumers).
    pub fn vote(&self) -> i32 {
        match self.producers.cmp(&self.consumers) {
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Less    => -1,
            std::cmp::Ordering::Equal   => 0,
        }
    }

    /// Net ternary charge: producers contribute +1, consumers −1, decomposers 0.
    pub fn charge(&self) -> i32 {
        self.producers as i32 - self.consumers as i32
    }
}

// ─── Ecosystem grid ───────────────────────────────────────────────────────────

/// The simulation grid. One step = competition + pheromone + warp consensus.
///
/// Key property: `mean_field = false` (default) restricts competition to
/// adjacent neighbors. Setting `mean_field = true` allows any cell to attack
/// any other, destroying the spatial structure that enables coexistence.
pub struct EcosystemGrid {
    cells: Vec<Cell>,
    pub width: usize,
    pub height: usize,
    pub step_count: u64,
    rng: Rng,
    /// When true, any cell can compete with any other cell — destroys coexistence.
    pub mean_field: bool,
}

impl EcosystemGrid {
    /// Uniform start: CAPACITY/3 of each species in every cell.
    pub fn new_uniform(width: usize, height: usize, seed: u64) -> Self {
        let each = CAPACITY / 3;
        EcosystemGrid {
            cells: vec![Cell::new(each, each, each); width * height],
            width, height, step_count: 0,
            rng: Rng::new(seed), mean_field: false,
        }
    }

    /// Random start: each cell gets a random partition of CAPACITY.
    pub fn new_random(width: usize, height: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let cells = (0..width * height).map(|_| {
            let p = rng.next_u32_max(CAPACITY + 1);
            let c = rng.next_u32_max(CAPACITY - p + 1);
            let d = CAPACITY - p - c;
            Cell::new(p, c, d)
        }).collect();
        EcosystemGrid {
            cells, width, height, step_count: 0,
            rng: Rng::new(seed ^ 0xC0FFEE), mean_field: false,
        }
    }

    fn idx(&self, x: usize, y: usize) -> usize { y * self.width + x }

    fn neighbors_of(&self, idx: usize) -> Vec<usize> {
        let (x, y) = (idx % self.width, idx / self.width);
        let mut nbrs = Vec::with_capacity(4);
        if x > 0              { nbrs.push(self.idx(x - 1, y)); }
        if x + 1 < self.width { nbrs.push(self.idx(x + 1, y)); }
        if y > 0              { nbrs.push(self.idx(x, y - 1)); }
        if y + 1 < self.height{ nbrs.push(self.idx(x, y + 1)); }
        nbrs
    }

    /// Advance one simulation tick.
    pub fn step(&mut self) {
        self.step_competition();
        self.step_pheromone();
        self.step_warp_consensus();
        self.step_count += 1;
    }

    /// Run `n` ticks.
    pub fn run(&mut self, n: u64) {
        for _ in 0..n { self.step(); }
    }

    // ── Competition phase ─────────────────────────────────────────────────────
    //
    // Each cell's dominant species "invades" one neighbor (or random cell in
    // mean-field mode). If the target has prey, one prey agent converts to
    // the aggressor species. Total in the target cell is unchanged (±0 net).

    fn step_competition(&mut self) {
        let n = self.cells.len();

        // Pre-compute neighbor lists (separated from rng usage to avoid borrow conflict)
        let all_nbrs: Vec<Vec<usize>> = (0..n).map(|i| self.neighbors_of(i)).collect();

        // Choose a target for each cell using the PRNG
        let targets: Vec<Option<usize>> = (0..n).map(|idx| {
            if self.mean_field {
                let t = self.rng.next_usize(n);
                if t != idx { Some(t) } else { None }
            } else {
                let nbrs = &all_nbrs[idx];
                if nbrs.is_empty() { return None; }
                Some(nbrs[self.rng.next_usize(nbrs.len())])
            }
        }).collect();

        // Randomize application order to reduce positional bias (Fisher-Yates)
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.rng.next_usize(i + 1);
            order.swap(i, j);
        }

        for &src in &order {
            let Some(tgt) = targets[src] else { continue };
            let aggressor = self.cells[src].dominant();
            let prey      = aggressor.beats();
            let prey_count = self.cells[tgt].get(prey);
            if prey_count > 0 {
                self.cells[tgt].add(prey,      -1);
                self.cells[tgt].add(aggressor,  1);
                // cells[tgt].total() is unchanged: one dies, one born ✓
            }
        }
    }

    // ── Pheromone phase ───────────────────────────────────────────────────────
    //
    // Each cell emits a signal proportional to its charge (producers − consumers).
    // Pheromone decays each step and bleeds into neighbors.
    // Result: producer-rich regions develop positive pheromone corridors.

    fn step_pheromone(&mut self) {
        let n = self.cells.len();

        // Decay
        for cell in &mut self.cells { cell.pheromone = (cell.pheromone as f32 * PHEROMONE_DECAY) as i32; }

        // Collect (emit + neighbor list) before mutating
        let signals: Vec<(i32, Vec<usize>)> = (0..n)
            .map(|i| (self.cells[i].charge() * PHEROMONE_EMISSION, self.neighbors_of(i)))
            .collect();

        for (idx, (emit, nbrs)) in signals.iter().enumerate() {
            self.cells[idx].pheromone += emit;
            let bleed = emit / 4; // 25% bleeds to each neighbor
            for &nbr in nbrs { self.cells[nbr].pheromone += bleed; }
        }
    }

    // ── Warp consensus phase ──────────────────────────────────────────────────
    //
    // Groups of WARP_SIZE cells vote. If ≥75% agree on a direction, the warp
    // reaches consensus and amplifies the winner by converting one opponent.
    // This simulates CUDA __ballot_sync + __reduce_add_sync.
    //
    // Conservation: one agent is converted (not created), so total is unchanged.

    fn step_warp_consensus(&mut self) {
        let n = self.cells.len();
        let num_warps = (n + WARP_SIZE - 1) / WARP_SIZE;

        for w in 0..num_warps {
            let start = w * WARP_SIZE;
            let end = (start + WARP_SIZE).min(n);
            let warp_len = (end - start) as i32;
            let votes: i32 = (start..end).map(|i| self.cells[i].vote()).sum();
            let threshold = warp_len * 3 / 4; // supermajority: ≥75%

            if votes >= threshold {
                // Producer supermajority → convert one consumer in the warp
                if let Some(tgt) = (start..end)
                    .filter(|&i| self.cells[i].consumers > 0)
                    .max_by_key(|&i| self.cells[i].consumers)
                {
                    self.cells[tgt].consumers  -= 1;
                    self.cells[tgt].producers  += 1;
                }
            } else if votes <= -threshold {
                // Consumer supermajority → convert one producer in the warp
                if let Some(tgt) = (start..end)
                    .filter(|&i| self.cells[i].producers > 0)
                    .max_by_key(|&i| self.cells[i].producers)
                {
                    self.cells[tgt].producers  -= 1;
                    self.cells[tgt].consumers  += 1;
                }
            }
        }
    }

    // ── Observations ─────────────────────────────────────────────────────────

    pub fn total_population(&self) -> u32 { self.cells.iter().map(|c| c.total()).sum() }

    pub fn species_counts(&self) -> (u32, u32, u32) {
        self.cells.iter().fold((0, 0, 0), |(p, c, d), cell| {
            (p + cell.producers, c + cell.consumers, d + cell.decomposers)
        })
    }

    pub fn total_charge(&self) -> i32 { self.cells.iter().map(|c| c.charge()).sum() }

    pub fn votes(&self) -> Vec<i32> { self.cells.iter().map(|c| c.vote()).collect() }

    pub fn pheromone_field(&self) -> Vec<i32> { self.cells.iter().map(|c| c.pheromone).collect() }

    pub fn cell(&self, x: usize, y: usize) -> &Cell { &self.cells[self.idx(x, y)] }

    /// Shannon entropy of global species proportions. Range: [0, ln(3) ≈ 1.099].
    /// High entropy = diverse ecosystem. Entropy collapses to 0 if one species wins.
    pub fn species_entropy(&self) -> f64 {
        let (p, c, d) = self.species_counts();
        let total = (p + c + d) as f64;
        if total == 0.0 { return 0.0; }
        [p, c, d].iter()
            .map(|&n| n as f64 / total)
            .filter(|&x| x > 0.0)
            .map(|x| -x * x.ln())
            .sum()
    }

    /// Moran's I proxy: spatial autocorrelation of votes.
    ///
    /// Near 0.0 = random (no clusters). Positive = like species cluster together.
    /// Emergent territory formation shows as rising autocorrelation over time.
    pub fn spatial_autocorrelation(&self) -> f64 {
        let n = self.cells.len();
        let mut xy = 0.0f64; let mut x2 = 0.0f64; let mut count = 0usize;
        for i in 0..n {
            let v = self.cells[i].vote() as f64;
            for nbr in self.neighbors_of(i) {
                let nv = self.cells[nbr].vote() as f64;
                xy += v * nv; x2 += v * v; count += 1;
            }
        }
        if count == 0 || x2 < 1e-10 { 0.0 } else { xy / (x2 / n as f64 * count as f64).sqrt() }
    }

    /// Number of cells where each species is strictly absent (count == 0).
    pub fn extinction_counts(&self) -> (usize, usize, usize) {
        self.cells.iter().fold((0, 0, 0), |(ep, ec, ed), cell| (
            ep + (cell.producers   == 0) as usize,
            ec + (cell.consumers   == 0) as usize,
            ed + (cell.decomposers == 0) as usize,
        ))
    }

    pub fn snapshot(&self) -> Vec<(u32, u32, u32)> {
        self.cells.iter().map(|c| (c.producers, c.consumers, c.decomposers)).collect()
    }
}

// ─── Warp consensus (standalone) ─────────────────────────────────────────────

/// Standalone warp vote reducer. Mirrors GPU warp-level reduction intrinsics.
///
/// Maps a flat slice of per-cell votes to per-warp consensus signals using
/// the same ≥75% supermajority threshold as `EcosystemGrid`.
pub struct WarpConsensus {
    pub warp_size: usize,
}

impl WarpConsensus {
    pub fn new(warp_size: usize) -> Self { WarpConsensus { warp_size } }

    /// Reduce votes to one signal per warp: +1, −1, or 0 (no consensus).
    pub fn reduce(&self, votes: &[i32]) -> Vec<i32> {
        votes.chunks(self.warp_size).map(|warp| {
            let sum: i32 = warp.iter().sum();
            let len = warp.len() as i32;
            if sum * 4 >= len * 3 { 1 }
            else if -sum * 4 >= len * 3 { -1 }
            else { 0 }
        }).collect()
    }

    /// Fraction of warps with a decisive vote.
    pub fn consensus_rate(&self, votes: &[i32]) -> f64 {
        let warp_votes = self.reduce(votes);
        let decisive = warp_votes.iter().filter(|&&v| v != 0).count();
        decisive as f64 / warp_votes.len().max(1) as f64
    }
}

// ─── PN-Counter CRDT ─────────────────────────────────────────────────────────

/// PN-Counter CRDT: tracks births and deaths per species across distributed nodes.
///
/// Each node has monotonically increasing birth and death counters.
/// Merge = max per-node, per-counter. Result: commutative, associative, idempotent.
///
/// Conservation invariant: in zero-sum competition (every death → one birth),
/// `total_net()` == 0 under any sequence of record + merge operations.
#[derive(Clone, Debug)]
pub struct PopCrdt {
    node_id: u32,
    // table[node] = [P_births, C_births, D_births, P_deaths, C_deaths, D_deaths]
    table: HashMap<u32, [u64; 6]>,
}

impl PopCrdt {
    pub fn new(node_id: u32) -> Self {
        let mut table = HashMap::new();
        table.insert(node_id, [0u64; 6]);
        PopCrdt { node_id, table }
    }

    pub fn record_birth(&mut self, s: Species) {
        self.table.entry(self.node_id).or_default()[s.birth_idx()] += 1;
    }

    pub fn record_death(&mut self, s: Species) {
        self.table.entry(self.node_id).or_default()[s.death_idx()] += 1;
    }

    /// Record a zero-sum competition: one winner born, one loser dies.
    pub fn record_competition(&mut self, winner: Species, loser: Species) {
        self.record_birth(winner);
        self.record_death(loser);
    }

    /// Merge another node's CRDT state. Idempotent, commutative, associative.
    pub fn merge(&mut self, other: &Self) {
        for (&node, &other_c) in &other.table {
            let c = self.table.entry(node).or_default();
            for i in 0..6 { c[i] = c[i].max(other_c[i]); }
        }
    }

    /// Net births per species summed across all nodes: (ΔP, ΔC, ΔD).
    pub fn net(&self) -> (i64, i64, i64) {
        self.table.values().fold((0i64, 0i64, 0i64), |(p, c, d), row| (
            p + row[0] as i64 - row[3] as i64,
            c + row[1] as i64 - row[4] as i64,
            d + row[2] as i64 - row[5] as i64,
        ))
    }

    /// Sum of all net changes. Must be 0 for zero-sum competition.
    pub fn total_net(&self) -> i64 { let (p, c, d) = self.net(); p + c + d }

    pub fn known_nodes(&self) -> usize { self.table.len() }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Z₃ algebra ───────────────────────────────────────────────────────────

    #[test]
    fn z3_beat_relation_is_a_cycle() {
        // P > D > C > P — one full cycle
        assert_eq!(Species::Producer.beats(),   Species::Decomposer);
        assert_eq!(Species::Decomposer.beats(), Species::Consumer);
        assert_eq!(Species::Consumer.beats(),   Species::Producer);
        // beaten_by is the inverse
        assert_eq!(Species::Producer.beaten_by(),   Species::Consumer);
        assert_eq!(Species::Decomposer.beaten_by(), Species::Producer);
        assert_eq!(Species::Consumer.beaten_by(),   Species::Decomposer);
        // The cycle has period 3: A.beats().beats().beats() == A
        let start = Species::Producer;
        assert_eq!(start.beats().beats().beats(), start);
    }

    #[test]
    fn z3_trit_values_are_canonical() {
        assert_eq!(Species::Producer.trit(),   1);
        assert_eq!(Species::Consumer.trit(),  -1);
        assert_eq!(Species::Decomposer.trit(), 0);
        // Z₃ charge: winner + loser always sums to loser's type
        // P(+1) + D(0) = +1  → winner (P) keeps its sign
        // D(0) + C(-1) = -1  → winner (D) doesn't; the CYCLE matters, not arithmetic
        // The conservation is topological, not additive
    }

    #[test]
    fn cell_capacity_is_invariant() {
        let cell = Cell::new(4, 4, 4);
        assert_eq!(cell.total(), CAPACITY);
        // Deliberate edge cases
        let all_p = Cell::new(CAPACITY, 0, 0);
        let all_c = Cell::new(0, CAPACITY, 0);
        let all_d = Cell::new(0, 0, CAPACITY);
        assert_eq!(all_p.total(), CAPACITY);
        assert_eq!(all_c.total(), CAPACITY);
        assert_eq!(all_d.total(), CAPACITY);
    }

    #[test]
    fn cell_dominant_reflects_plurality() {
        let mostly_producers = Cell::new(8, 2, 2);
        assert_eq!(mostly_producers.dominant(), Species::Producer);

        let mostly_consumers = Cell::new(2, 8, 2);
        assert_eq!(mostly_consumers.dominant(), Species::Consumer);

        let mostly_decomposers = Cell::new(2, 2, 8);
        assert_eq!(mostly_decomposers.dominant(), Species::Decomposer);

        // Tie: Producer wins (deterministic tiebreak)
        let tied = Cell::new(4, 4, 4);
        assert_eq!(tied.dominant(), Species::Producer);
    }

    #[test]
    fn cell_vote_is_sign_of_charge() {
        let pro_heavy = Cell::new(8, 2, 2);
        assert_eq!(pro_heavy.vote(), 1);
        assert_eq!(pro_heavy.charge(), 6);

        let con_heavy = Cell::new(2, 8, 2);
        assert_eq!(con_heavy.vote(), -1);
        assert_eq!(con_heavy.charge(), -6);

        let balanced = Cell::new(4, 4, 4);
        assert_eq!(balanced.vote(), 0);
        assert_eq!(balanced.charge(), 0);
    }

    // ── Conservation law ─────────────────────────────────────────────────────

    #[test]
    fn competition_conserves_total_population_exactly() {
        let mut grid = EcosystemGrid::new_random(12, 12, 0xBEEF);
        let initial = grid.total_population();
        grid.run(200);
        assert_eq!(grid.total_population(), initial,
            "total agents must never change: every death pairs with a birth");
    }

    #[test]
    fn competition_conserves_per_cell_capacity() {
        let mut grid = EcosystemGrid::new_random(8, 8, 0xCAFE);
        grid.run(150);
        for y in 0..grid.height {
            for x in 0..grid.width {
                let cell = grid.cell(x, y);
                assert_eq!(cell.total(), CAPACITY,
                    "cell ({},{}) has {} agents, expected {}", x, y, cell.total(), CAPACITY);
            }
        }
    }

    // ── Emergent behavior: spatial coexistence ────────────────────────────────

    #[test]
    fn spatial_rps_maintains_all_three_species() {
        // Key claim: local interactions → coexistence emerges.
        // With spatial structure, no single species can eliminate the others.
        let mut grid = EcosystemGrid::new_random(12, 12, 0x5EED);
        grid.run(400);
        let (p, c, d) = grid.species_counts();
        let total = (p + c + d) as f64;
        let p_frac = p as f64 / total;
        let c_frac = c as f64 / total;
        let d_frac = d as f64 / total;
        assert!(p_frac > 0.05, "producers went locally extinct ({:.1}%)", p_frac * 100.0);
        assert!(c_frac > 0.05, "consumers went locally extinct ({:.1}%)", c_frac * 100.0);
        assert!(d_frac > 0.05, "decomposers went locally extinct ({:.1}%)", d_frac * 100.0);
    }

    #[test]
    fn mean_field_collapses_diversity_compared_to_spatial() {
        // Coexistence depends on LOCAL interactions.
        // Mean-field mode (any cell vs any cell) destroys the spatial protection.
        let seed = 0x1234_5678u64;
        let steps = 300u64;

        let mut spatial = EcosystemGrid::new_random(10, 10, seed);
        spatial.run(steps);
        let spatial_entropy = spatial.species_entropy();

        let mut mf = EcosystemGrid::new_random(10, 10, seed);
        mf.mean_field = true;
        mf.run(steps);
        let mf_entropy = mf.species_entropy();

        assert!(
            spatial_entropy > mf_entropy + 0.05,
            "spatial entropy {:.3} should exceed mean-field {:.3} \
             (spatial structure creates coexistence that mean-field destroys)",
            spatial_entropy, mf_entropy
        );
    }

    #[test]
    fn species_entropy_stays_high_in_spatial_mode() {
        // Maximum entropy for 3 equal species is ln(3) ≈ 1.099.
        // After 400 steps, spatial coexistence should keep entropy > 0.75.
        let mut grid = EcosystemGrid::new_random(14, 14, 0xABCD);
        grid.run(400);
        let entropy = grid.species_entropy();
        assert!(entropy > 0.75,
            "entropy {:.3} too low — ecosystem lost diversity (max is ln(3)≈1.099)",
            entropy);
    }

    // ── Pheromone communication ───────────────────────────────────────────────

    #[test]
    fn pheromone_sign_tracks_local_species_balance() {
        // After running, cells with more producers than consumers should have
        // positive pheromone, and vice versa.
        let mut grid = EcosystemGrid::new_random(10, 10, 0xF00D);
        grid.run(100);

        let mut positive_matches = 0usize;
        let mut negative_matches = 0usize;
        let mut total_nonzero = 0usize;

        for y in 0..grid.height {
            for x in 0..grid.width {
                let cell = grid.cell(x, y);
                if cell.pheromone != 0 || cell.charge() != 0 {
                    total_nonzero += 1;
                    if cell.charge() > 0 && cell.pheromone > 0 { positive_matches += 1; }
                    if cell.charge() < 0 && cell.pheromone < 0 { negative_matches += 1; }
                }
            }
        }
        // More than half of non-zero cells should have matching sign
        let matches = positive_matches + negative_matches;
        assert!(matches * 2 >= total_nonzero,
            "pheromone sign should correlate with charge: {}/{} matched",
            matches, total_nonzero);
    }

    #[test]
    fn pheromone_decays_to_near_zero_without_source() {
        let mut grid = EcosystemGrid::new_random(6, 6, 0x1111);
        // Seed pheromone without running competition
        for cell in &mut grid.cells { cell.pheromone = 1000; }
        // Zero out all agents except decomposers (zero charge → zero emission)
        for cell in &mut grid.cells {
            cell.consumers = 0; cell.producers = 0; cell.decomposers = CAPACITY;
        }
        // 30 pheromone steps: 0.82^30 ≈ 0.004 → should be near 0
        for _ in 0..30 { grid.step_pheromone(); }
        let max_pheromone = grid.pheromone_field().into_iter().map(|v| v.abs()).max().unwrap_or(0);
        assert!(max_pheromone < 10, "pheromone should decay; max residual = {}", max_pheromone);
    }

    // ── Warp consensus ────────────────────────────────────────────────────────

    #[test]
    fn warp_consensus_requires_supermajority() {
        let wc = WarpConsensus::new(4);

        // 4/4 producers → strong +1 consensus
        assert_eq!(wc.reduce(&[1, 1, 1, 1]),  vec![1]);
        // 3/4 producers (75% = threshold) → consensus
        assert_eq!(wc.reduce(&[1, 1, 1, -1]), vec![1]);
        // 2/4 producers (50% = no consensus)
        assert_eq!(wc.reduce(&[1, 1, -1, -1]), vec![0]);
        // 4/4 consumers → −1
        assert_eq!(wc.reduce(&[-1, -1, -1, -1]), vec![-1]);
        // Mixed across two warps
        let votes = [1, 1, 1, 1,   -1, -1, -1, -1];
        assert_eq!(wc.reduce(&votes), vec![1, -1]);
    }

    #[test]
    fn warp_consensus_rate_rises_after_clustering() {
        // Initially random → low consensus.
        // After running, clusters form → nearby cells agree → higher consensus rate.
        let mut grid = EcosystemGrid::new_random(12, 12, 0x9999);
        let wc = WarpConsensus::new(WARP_SIZE);

        let rate_before = wc.consensus_rate(&grid.votes());
        grid.run(300);
        let rate_after = wc.consensus_rate(&grid.votes());

        assert!(
            rate_after > rate_before,
            "warp consensus rate should rise as clusters form: {:.2} → {:.2}",
            rate_before, rate_after
        );
    }

    // ── Emergent spatial clustering ───────────────────────────────────────────

    #[test]
    fn spatial_autocorrelation_rises_as_territories_form() {
        // Random start: autocorrelation ≈ 0 (no structure).
        // After running: positive autocorrelation (like species cluster).
        // This is the spatial signature of emergent territory formation.
        let mut grid = EcosystemGrid::new_random(14, 14, 0x4444);
        let ac_before = grid.spatial_autocorrelation();
        grid.run(350);
        let ac_after = grid.spatial_autocorrelation();

        assert!(
            ac_after > ac_before + 0.02,
            "autocorrelation should rise as territories form: {:.4} → {:.4}",
            ac_before, ac_after
        );
    }

    // ── CRDT laws ─────────────────────────────────────────────────────────────

    #[test]
    fn crdt_merge_is_commutative() {
        let mut a = PopCrdt::new(0);
        a.record_competition(Species::Producer, Species::Decomposer);
        a.record_competition(Species::Producer, Species::Decomposer);

        let mut b = PopCrdt::new(1);
        b.record_competition(Species::Consumer, Species::Producer);

        // a ⊔ b
        let mut ab = a.clone();
        ab.merge(&b);

        // b ⊔ a
        let mut ba = b.clone();
        ba.merge(&a);

        // Both should have the same totals
        assert_eq!(ab.net(), ba.net(), "merge must be commutative: a⊔b = b⊔a");
    }

    #[test]
    fn crdt_merge_is_idempotent() {
        let mut a = PopCrdt::new(0);
        a.record_competition(Species::Decomposer, Species::Consumer);
        a.record_competition(Species::Decomposer, Species::Consumer);

        let snapshot = a.clone();
        a.merge(&snapshot);
        a.merge(&snapshot); // double-merge

        // Net should be unchanged from original
        assert_eq!(a.net(), snapshot.net(), "merge must be idempotent: a⊔a = a");
    }

    #[test]
    fn crdt_preserves_zero_sum_conservation() {
        // Every competition is zero-sum: one birth, one death.
        // total_net() must always be 0.
        let mut node0 = PopCrdt::new(0);
        let mut node1 = PopCrdt::new(1);

        // Simulate independent competition on two nodes
        for _ in 0..50 {
            node0.record_competition(Species::Producer,   Species::Decomposer);
            node0.record_competition(Species::Consumer,   Species::Producer);
            node1.record_competition(Species::Decomposer, Species::Consumer);
        }

        assert_eq!(node0.total_net(), 0, "node0: every birth paired with death");
        assert_eq!(node1.total_net(), 0, "node1: every birth paired with death");

        // After merge, total_net still 0
        let mut merged = node0.clone();
        merged.merge(&node1);
        assert_eq!(merged.total_net(), 0,
            "merged CRDT: conservation holds across nodes after merge");
    }

    #[test]
    fn crdt_merge_combines_node_knowledge() {
        let mut a = PopCrdt::new(10);
        a.record_competition(Species::Producer, Species::Decomposer);

        let mut b = PopCrdt::new(20);
        b.record_competition(Species::Consumer, Species::Producer);

        assert_eq!(a.known_nodes(), 1);
        assert_eq!(b.known_nodes(), 1);

        a.merge(&b);
        assert_eq!(a.known_nodes(), 2, "after merge, a knows about b's node");
    }

    // ── End-to-end emergent ecosystem scenario ────────────────────────────────

    #[test]
    fn full_ecosystem_run_shows_all_emergent_properties() {
        let mut grid = EcosystemGrid::new_random(16, 16, 0xDEAD_BEEF);
        let wc = WarpConsensus::new(WARP_SIZE);

        let pop_before = grid.total_population();
        let entropy_before = grid.species_entropy();
        let ac_before = grid.spatial_autocorrelation();
        let consensus_before = wc.consensus_rate(&grid.votes());

        grid.run(500);

        let pop_after = grid.total_population();
        let entropy_after = grid.species_entropy();
        let ac_after = grid.spatial_autocorrelation();
        let consensus_after = wc.consensus_rate(&grid.votes());

        // 1. Conservation: total population unchanged
        assert_eq!(pop_before, pop_after, "conservation law violated");

        // 2. Coexistence: all species still present
        let (p, c, d) = grid.species_counts();
        assert!(p > 0, "producers went extinct");
        assert!(c > 0, "consumers went extinct");
        assert!(d > 0, "decomposers went extinct");

        // 3. Diversity maintained (entropy ≥ 0.6)
        assert!(entropy_after >= 0.60,
            "ecosystem lost diversity: entropy = {:.3}", entropy_after);

        // 4. Spatial structure formed: autocorrelation rose
        assert!(ac_after > ac_before,
            "no spatial clustering detected: ac = {:.4} → {:.4}", ac_before, ac_after);

        // 5. Warp consensus increased: warps became more opinionated
        assert!(consensus_after > consensus_before,
            "warp consensus did not increase: {:.2} → {:.2}", consensus_before, consensus_after);
    }
}
