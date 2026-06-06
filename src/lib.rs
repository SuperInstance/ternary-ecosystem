//! # Ternary Ecosystem
//!
//! A ternary ecosystem simulation where organisms are classified by their role
//! in the food chain using trit values: Predator (-1), Plant (0), and Prey (+1).
//!
//! The ecosystem runs on a spatial grid where interactions happen between
//! neighboring cells. The balance between these three roles creates emergent
//! population dynamics — predator-prey oscillations, plant growth cycles,
//! and carrying capacity limits — all from simple ternary rules.
//!
//! ## Core Model
//!
//! ```text
//! Predator (-1): eats prey, dies without food
//! Plant    ( 0): grows each tick, eaten by prey
//! Prey    (+1): eats plants, eaten by predators, reproduces when well-fed
//! ```

use std::fmt;

/// A ternary organism role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Organism {
    Predator = -1,
    Plant = 0,
    Prey = 1,
}

impl Organism {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Organism::Predator),
            0 => Some(Organism::Plant),
            1 => Some(Organism::Prey),
            _ => None,
        }
    }

    pub fn trit(self) -> i8 {
        self as i8
    }
}

impl fmt::Display for Organism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Organism::Predator => write!(f, "Predator(-1)"),
            Organism::Plant => write!(f, "Plant(0)"),
            Organism::Prey => write!(f, "Prey(+1)"),
        }
    }
}

/// Population counts for a cell.
#[derive(Debug, Clone, Default)]
pub struct CellPopulation {
    pub predators: u32,
    pub plants: u32,
    pub prey: u32,
}

impl CellPopulation {
    pub fn new(predators: u32, plants: u32, prey: u32) -> Self {
        Self { predators, plants, prey }
    }

    pub fn total(&self) -> u32 {
        self.predators + self.plants + self.prey
    }

    pub fn dominant(&self) -> Organism {
        if self.predators >= self.plants && self.predators >= self.prey {
            Organism::Predator
        } else if self.prey >= self.plants {
            Organism::Prey
        } else {
            Organism::Plant
        }
    }
}

/// Simple deterministic PRNG.
#[derive(Clone, Debug)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 1 } else { seed })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_u32(&mut self, n: u32) -> u32 {
        (self.next() as u32) % n
    }
}

/// Configuration parameters for the ecosystem simulation.
#[derive(Debug, Clone)]
pub struct EcosystemConfig {
    /// Maximum population per cell (carrying capacity).
    pub cell_capacity: u32,
    /// Plant growth per tick (absolute number added).
    pub plant_growth: u32,
    /// Predation efficiency: fraction of predator-prey encounters resulting in a kill.
    pub predation_efficiency: f64,
    /// Grazing efficiency: fraction of prey-plant encounters.
    pub grazing_efficiency: f64,
    /// Predator starvation: fraction dying per tick when unfed.
    pub predator_starvation: f64,
    /// Prey birth rate per well-fed prey.
    pub prey_birth_rate: f64,
    /// Predator birth rate per well-fed predator.
    pub predator_birth_rate: f64,
}

impl Default for EcosystemConfig {
    fn default() -> Self {
        Self {
            cell_capacity: 100,
            plant_growth: 10,
            predation_efficiency: 0.4,
            grazing_efficiency: 0.3,
            predator_starvation: 0.1,
            prey_birth_rate: 0.4,
            predator_birth_rate: 0.25,
        }
    }
}

/// A spatial ecosystem grid with ternary organism roles.
#[derive(Debug, Clone)]
pub struct EcosystemGrid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<CellPopulation>,
    pub config: EcosystemConfig,
    rng: Rng,
    step_count: u64,
}

impl EcosystemGrid {
    /// Create a new grid with all zeros.
    pub fn new(width: usize, height: usize, seed: u64) -> Self {
        let cells = vec![CellPopulation::default(); width * height];
        Self {
            width,
            height,
            cells,
            config: EcosystemConfig::default(),
            rng: Rng::new(seed),
            step_count: 0,
        }
    }

    /// Create a grid with random initial populations.
    pub fn new_random(width: usize, height: usize, seed: u64) -> Self {
        let mut grid = Self::new(width, height, seed);
        let cap = grid.config.cell_capacity;
        for cell in &mut grid.cells {
            cell.plants = 20 + grid.rng.next_u32(cap / 2);
            cell.prey = 5 + grid.rng.next_u32(cap / 6);
            cell.predators = 2 + grid.rng.next_u32(cap / 10);
        }
        grid
    }

    /// Get a cell by (x, y).
    pub fn cell(&self, x: usize, y: usize) -> &CellPopulation {
        &self.cells[y * self.width + x]
    }

    /// Get a mutable cell by (x, y).
    pub fn cell_mut(&mut self, x: usize, y: usize) -> &mut CellPopulation {
        &mut self.cells[y * self.width + x]
    }

    /// Get indices of the 8-connected neighbors of a cell.
    fn neighbors(&self, idx: usize) -> Vec<usize> {
        let x = idx % self.width;
        let y = idx / self.width;
        let mut nbrs = Vec::new();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    nbrs.push((ny as usize) * self.width + (nx as usize));
                }
            }
        }
        nbrs
    }

    /// Run the simulation for `n` steps.
    pub fn run(&mut self, n: u64) {
        for _ in 0..n { self.step(); }
    }

    /// Advance the simulation by one step using Lotka-Volterra dynamics.
    pub fn step(&mut self) {
        let n = self.cells.len();
        let cap = self.config.cell_capacity;

        // Snapshot for synchronous update
        let snap: Vec<[u32; 3]> = self.cells.iter()
            .map(|c| [c.predators, c.plants, c.prey])
            .collect();

        for idx in 0..n {
            let [mut pred, mut plant, mut prey] = snap[idx];

            // 1. Plant growth (logistic toward capacity)
            let grow = self.config.plant_growth.min(cap - plant);
            plant = plant.saturating_add(grow).min(cap);

            // 2. Grazing: prey * plants * efficiency / capacity (Lotka-Volterra interaction term)
            let grazed = (self.config.grazing_efficiency * prey as f64 * plant as f64 / cap as f64).ceil() as u32;
            let grazed = grazed.min(plant).min(prey); // Can't eat more plants than exist or than prey
            plant = plant.saturating_sub(grazed);

            // 3. Predation: predators * prey * efficiency / capacity
            let hunted = (self.config.predation_efficiency * pred as f64 * prey as f64 / cap as f64).ceil() as u32;
            let hunted = hunted.min(prey);
            prey = prey.saturating_sub(hunted);

            // 4. Prey reproduction: proportional to grazing success
            if prey > 0 && grazed > 0 {
                let fed_fraction = (grazed as f64 / prey as f64).min(1.0);
                let births = (self.config.prey_birth_rate * prey as f64 * fed_fraction).ceil() as u32;
                prey = prey.saturating_add(births).min(cap);
            }

            // 5. Predator reproduction: proportional to hunting success
            if pred > 0 && hunted > 0 {
                let fed_fraction = (hunted as f64 / pred as f64).min(1.0);
                let births = (self.config.predator_birth_rate * pred as f64 * fed_fraction).ceil() as u32;
                pred = pred.saturating_add(births).min(cap);
            }

            // 6. Predator starvation when they didn't eat enough
            if pred > 0 {
                let needed_food = (pred as f64 * self.config.predation_efficiency).ceil() as u32;
                if hunted < needed_food.saturating_add(1) / 2 {
                    let starved = (self.config.predator_starvation * pred as f64).max(1.0) as u32;
                    pred = pred.saturating_sub(starved);
                }
            }

            self.cells[idx].predators = pred.min(cap);
            self.cells[idx].plants = plant.min(cap);
            self.cells[idx].prey = prey.min(cap);
        }

        self.step_count += 1;
    }

    /// Total population across all cells.
    pub fn total_population(&self) -> u32 {
        self.cells.iter().map(|c| c.total()).sum()
    }

    /// Count each species globally.
    pub fn species_counts(&self) -> (u32, u32, u32) {
        self.cells.iter().fold((0, 0, 0), |(p, pl, pr), c| {
            (p + c.predators, pl + c.plants, pr + c.prey)
        })
    }

    /// Check if a species is extinct.
    pub fn is_extinct(&self, organism: Organism) -> bool {
        match organism {
            Organism::Predator => self.cells.iter().all(|c| c.predators == 0),
            Organism::Plant => self.cells.iter().all(|c| c.plants == 0),
            Organism::Prey => self.cells.iter().all(|c| c.prey == 0),
        }
    }

    /// Food chain depth (0-3 trophic levels with non-zero population).
    pub fn food_chain_depth(&self) -> usize {
        let (pred, plant, prey) = self.species_counts();
        let mut d = 0;
        if pred > 0 { d += 1; }
        if plant > 0 { d += 1; }
        if prey > 0 { d += 1; }
        d
    }

    /// Shannon entropy of species distribution (max = ln(3) ≈ 1.099).
    pub fn stability_metric(&self) -> f64 {
        let (pred, plant, prey) = self.species_counts();
        let total = (pred + plant + prey) as f64;
        if total == 0.0 { return 0.0; }
        let mut h = 0.0;
        for &c in &[pred, plant, prey] {
            if c > 0 {
                let p = c as f64 / total;
                h -= p * p.ln();
            }
        }
        h
    }

    /// Current step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

/// Tracks population dynamics over time.
#[derive(Debug, Clone, Default)]
pub struct PopulationTracker {
    pub history: Vec<(u32, u32, u32)>,
}

impl PopulationTracker {
    pub fn new() -> Self { Self { history: Vec::new() } }

    pub fn record(&mut self, predators: u32, plants: u32, prey: u32) {
        self.history.push((predators, plants, prey));
    }

    /// Detect predator-prey oscillation in recorded history.
    pub fn detect_oscillation(&self) -> bool {
        if self.history.len() < 10 { return false; }
        let prey_counts: Vec<u32> = self.history.iter().map(|h| h.2).collect();
        let pred_counts: Vec<u32> = self.history.iter().map(|h| h.0).collect();
        has_oscillation(&prey_counts) && has_oscillation(&pred_counts)
    }

    /// Average populations over history.
    pub fn average_populations(&self) -> (f64, f64, f64) {
        if self.history.is_empty() { return (0.0, 0.0, 0.0); }
        let n = self.history.len() as f64;
        let (sp, spl, spr) = self.history.iter()
            .fold((0u64, 0u64, 0u64), |(a, b, c), (p, pl, pr)| {
                (a + *p as u64, b + *pl as u64, c + *pr as u64)
            });
        (sp as f64 / n, spl as f64 / n, spr as f64 / n)
    }
}

fn has_oscillation(counts: &[u32]) -> bool {
    if counts.len() < 4 { return false; }
    let (mut inc, mut dec) = (0, 0);
    for i in 1..counts.len() {
        if counts[i] > counts[i - 1] { inc += 1; }
        else if counts[i] < counts[i - 1] { dec += 1; }
    }
    inc >= 2 && dec >= 2
}

/// Carrying capacity calculator.
pub struct CarryingCapacity;

impl CarryingCapacity {
    pub fn for_prey(config: &EcosystemConfig, grid_cells: usize) -> u32 {
        (config.plant_growth * grid_cells as u32) / 2
    }

    pub fn for_predators(config: &EcosystemConfig, grid_cells: usize) -> u32 {
        Self::for_prey(config, grid_cells) / 5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_organism_trit_values() {
        assert_eq!(Organism::Predator.trit(), -1);
        assert_eq!(Organism::Plant.trit(), 0);
        assert_eq!(Organism::Prey.trit(), 1);
    }

    #[test]
    fn test_organism_from_i8() {
        assert_eq!(Organism::from_i8(-1), Some(Organism::Predator));
        assert_eq!(Organism::from_i8(0), Some(Organism::Plant));
        assert_eq!(Organism::from_i8(1), Some(Organism::Prey));
        assert_eq!(Organism::from_i8(2), None);
    }

    #[test]
    fn test_organism_display() {
        assert!(format!("{}", Organism::Predator).contains("Predator"));
        assert!(format!("{}", Organism::Plant).contains("Plant"));
        assert!(format!("{}", Organism::Prey).contains("Prey"));
    }

    #[test]
    fn test_cell_population() {
        let cell = CellPopulation::new(10, 20, 15);
        assert_eq!(cell.total(), 45);
        assert_eq!(cell.dominant(), Organism::Plant);
    }

    #[test]
    fn test_cell_dominant() {
        assert_eq!(CellPopulation::new(30, 10, 5).dominant(), Organism::Predator);
        assert_eq!(CellPopulation::new(5, 10, 30).dominant(), Organism::Prey);
        assert_eq!(CellPopulation::new(5, 30, 10).dominant(), Organism::Plant);
    }

    #[test]
    fn test_grid_new() {
        let grid = EcosystemGrid::new(5, 5, 42);
        assert_eq!(grid.width, 5);
        assert_eq!(grid.height, 5);
        assert_eq!(grid.cells.len(), 25);
        assert_eq!(grid.total_population(), 0);
    }

    #[test]
    fn test_grid_new_random() {
        let grid = EcosystemGrid::new_random(4, 4, 42);
        assert!(grid.total_population() > 0);
    }

    #[test]
    fn test_grid_cell_access() {
        let mut grid = EcosystemGrid::new(3, 3, 42);
        grid.cell_mut(1, 1).predators = 5;
        assert_eq!(grid.cell(1, 1).predators, 5);
    }

    #[test]
    fn test_plant_growth() {
        let mut grid = EcosystemGrid::new(3, 3, 42);
        grid.cell_mut(1, 1).plants = 10;
        grid.cell_mut(1, 1).prey = 0;
        grid.cell_mut(1, 1).predators = 0;
        let before = grid.cell(1, 1).plants;
        grid.step();
        assert!(grid.cell(1, 1).plants > before);
    }

    #[test]
    fn test_plant_growth_respects_capacity() {
        let mut grid = EcosystemGrid::new(3, 3, 42);
        grid.cell_mut(1, 1).plants = grid.config.cell_capacity;
        grid.cell_mut(1, 1).prey = 0;
        grid.cell_mut(1, 1).predators = 0;
        grid.step();
        assert!(grid.cell(1, 1).plants <= grid.config.cell_capacity);
    }

    #[test]
    fn test_predator_starvation() {
        let mut grid = EcosystemGrid::new(3, 3, 42);
        for cell in &mut grid.cells {
            cell.predators = 20;
            cell.plants = 0;
            cell.prey = 0;
        }
        let before = grid.cell(1, 1).predators;
        grid.step();
        assert!(grid.cell(1, 1).predators < before);
    }

    #[test]
    fn test_predator_prey_oscillation() {
        let mut grid = EcosystemGrid::new_random(10, 10, 12345);
        let mut tracker = PopulationTracker::new();
        for _ in 0..300 {
            grid.step();
            let (p, pl, pr) = grid.species_counts();
            tracker.record(p, pl, pr);
        }
        assert!(tracker.detect_oscillation(),
            "predator-prey oscillation should be detectable");
    }

    #[test]
    fn test_all_species_survive() {
        let mut grid = EcosystemGrid::new_random(10, 10, 99999);
        grid.run(300);
        assert!(!grid.is_extinct(Organism::Predator), "predators should survive");
        assert!(!grid.is_extinct(Organism::Plant), "plants should survive");
        assert!(!grid.is_extinct(Organism::Prey), "prey should survive");
    }

    #[test]
    fn test_food_chain_depth() {
        let mut grid = EcosystemGrid::new_random(10, 10, 42);
        grid.run(50);
        assert_eq!(grid.food_chain_depth(), 3);
    }

    #[test]
    fn test_stability_metric_range() {
        let grid = EcosystemGrid::new_random(8, 8, 42);
        let s = grid.stability_metric();
        assert!(s >= 0.0);
        assert!(s <= (3.0_f64).ln() + 0.01);
    }

    #[test]
    fn test_stability_metric_equal_pops() {
        let mut grid = EcosystemGrid::new(2, 2, 42);
        for cell in &mut grid.cells {
            cell.predators = 10;
            cell.plants = 10;
            cell.prey = 10;
        }
        let s = grid.stability_metric();
        assert!((s - (3.0_f64).ln()).abs() < 0.001);
    }

    #[test]
    fn test_extinction_detection() {
        let mut grid = EcosystemGrid::new(2, 2, 42);
        for cell in &mut grid.cells { cell.plants = 20; }
        assert!(grid.is_extinct(Organism::Predator));
        assert!(grid.is_extinct(Organism::Prey));
        assert!(!grid.is_extinct(Organism::Plant));
    }

    #[test]
    fn test_population_tracker() {
        let mut t = PopulationTracker::new();
        t.record(5, 20, 10);
        t.record(3, 25, 8);
        t.record(4, 22, 12);
        assert_eq!(t.history.len(), 3);
        let (a, b, c) = t.average_populations();
        assert!((a - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_oscillation_detection() {
        let mut t = PopulationTracker::new();
        for i in 0..20 {
            let v = 100 + (50.0 * (i as f64 * 0.5).sin()) as u32;
            t.record(v, 200, v);
        }
        assert!(t.detect_oscillation());
    }

    #[test]
    fn test_no_oscillation_flat() {
        let mut t = PopulationTracker::new();
        for _ in 0..20 { t.record(10, 20, 10); }
        assert!(!t.detect_oscillation());
    }

    #[test]
    fn test_carrying_capacity() {
        let config = EcosystemConfig::default();
        let cap_prey = CarryingCapacity::for_prey(&config, 64);
        let cap_pred = CarryingCapacity::for_predators(&config, 64);
        assert!(cap_prey > 0);
        assert!(cap_pred > 0);
        assert!(cap_pred < cap_prey);
    }

    #[test]
    fn test_step_count() {
        let mut grid = EcosystemGrid::new(3, 3, 42);
        assert_eq!(grid.step_count(), 0);
        grid.run(10);
        assert_eq!(grid.step_count(), 10);
    }

    #[test]
    fn test_neighbors() {
        let grid = EcosystemGrid::new(5, 5, 42);
        assert_eq!(grid.neighbors(0).len(), 3); // corner
        assert_eq!(grid.neighbors(12).len(), 8); // center
    }

    #[test]
    fn test_ternary_population_balance() {
        let mut grid = EcosystemGrid::new_random(10, 10, 77777);
        grid.run(100);
        let (pred, plant, prey) = grid.species_counts();
        assert!(plant > 0, "plants should exist");
        assert!(prey > 0, "prey should exist");
        // Ternary pyramid: producers > consumers > apex
        assert!(plant > pred, "plants ({}) > predators ({})", plant, pred);
    }

    #[test]
    fn test_large_grid_stability() {
        let mut grid = EcosystemGrid::new_random(20, 20, 31415);
        grid.run(500);
        assert!(grid.stability_metric() > 0.3);
        assert_eq!(grid.food_chain_depth(), 3);
    }

    #[test]
    fn test_predator_extinction_without_prey() {
        let mut grid = EcosystemGrid::new(3, 3, 42);
        for cell in &mut grid.cells {
            cell.predators = 20;
            cell.plants = 30;
            cell.prey = 0;
        }
        grid.run(200);
        assert!(grid.is_extinct(Organism::Predator));
    }

    #[test]
    fn test_prey_grows_without_predators() {
        let mut grid = EcosystemGrid::new(5, 5, 42);
        for cell in &mut grid.cells {
            cell.prey = 5;
            cell.plants = 50;
            cell.predators = 0;
        }
        let initial: u32 = grid.cells.iter().map(|c| c.prey).sum();
        grid.run(50);
        let final_count: u32 = grid.cells.iter().map(|c| c.prey).sum();
        assert!(final_count > initial, "prey should grow: {} -> {}", initial, final_count);
    }
}

#[cfg(test)]
mod debug2 {
    use super::*;
    #[test]
    fn debug_pred_no_prey() {
        let mut grid = EcosystemGrid::new(3, 3, 42);
        for cell in &mut grid.cells {
            cell.predators = 20;
            cell.plants = 30;
            cell.prey = 0;
        }
        for i in 0..20 {
            grid.step();
            let (p, pl, pr) = grid.species_counts();
            if i % 5 == 0 || p == 0 {
                eprintln!("step {}: pred={} plant={} prey={}", i+1, p, pl, pr);
            }
        }
    }
}
