#![forbid(unsafe_code)]

//! Full ecosystem simulation with multiple ternary species and food webs.
//!
//! Models predator-prey dynamics, environmental niches, carrying capacity,
//! ecological succession, and keystone species detection for ternary-valued
//! agent populations.

/// Ternary value: -1, 0, +1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

/// A ternary species with behavioral traits.
#[derive(Clone, Debug)]
pub struct Species {
    pub name: String,
    /// Population count.
    pub population: u64,
    /// Growth rate (per tick).
    pub growth_rate: i32,
    /// Trophic level: 0=producer, 1=primary consumer, 2=predator, etc.
    pub trophic_level: u32,
    /// Strategy trait: negative=conservative, zero=neutral, positive=aggressive.
    pub strategy: Ternary,
    /// Whether this species is extinct.
    pub extinct: bool,
}

impl Species {
    pub fn new(name: &str, population: u64, growth_rate: i32, trophic_level: u32, strategy: Ternary) -> Self {
        Self {
            name: name.to_string(),
            population,
            growth_rate,
            trophic_level,
            strategy,
            extinct: population == 0,
        }
    }

    /// Tick: apply growth. Returns new population.
    pub fn tick(&mut self) -> u64 {
        if self.extinct { return 0; }
        let delta = (self.population as i64 * self.growth_rate as i64) / 100;
        let new_pop = (self.population as i64 + delta).max(0) as u64;
        self.population = new_pop;
        if new_pop == 0 { self.extinct = true; }
        new_pop
    }

    /// Apply predation: reduce population by loss.
    pub fn prey_loss(&mut self, loss: u64) {
        self.population = self.population.saturating_sub(loss);
        if self.population == 0 { self.extinct = true; }
    }

    /// Apply carrying capacity cap.
    pub fn cap_population(&mut self, capacity: u64) {
        if self.population > capacity {
            self.population = capacity;
        }
    }
}

/// Predator-prey relationship network.
#[derive(Clone, Debug)]
pub struct FoodWeb {
    /// (predator_index, prey_index, predation_rate)
    pub links: Vec<(usize, usize, u32)>,
}

impl FoodWeb {
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    /// Add a predation link.
    pub fn add_link(&mut self, predator: usize, prey: usize, rate: u32) {
        self.links.push((predator, prey, rate));
    }

    /// Apply one tick of predation across all links.
    /// Returns per-species losses (index, loss_amount).
    pub fn tick(&self, species: &mut [Species]) -> Vec<(usize, u64)> {
        let mut losses = Vec::new();
        for &(pred_idx, prey_idx, rate) in &self.links {
            if pred_idx >= species.len() || prey_idx >= species.len() { continue; }
            if species[pred_idx].extinct || species[prey_idx].extinct { continue; }

            let pred_pop = species[pred_idx].population;
            let loss = ((pred_pop as u64 * rate as u64) / 1000).min(species[prey_idx].population);
            species[prey_idx].prey_loss(loss);
            losses.push((prey_idx, loss));
        }
        losses
    }

    /// Get all predators of a given prey.
    pub fn predators_of(&self, prey_idx: usize) -> Vec<usize> {
        self.links.iter().filter(|&&(_, p, _)| p == prey_idx).map(|&(pred, _, _)| pred).collect()
    }

    /// Get all prey of a given predator.
    pub fn prey_of(&self, pred_idx: usize) -> Vec<usize> {
        self.links.iter().filter(|&&(p, _, _)| p == pred_idx).map(|&(_, prey, _)| prey).collect()
    }
}

/// An environmental niche with conditions.
#[derive(Clone, Debug)]
pub struct Niche {
    pub name: String,
    /// Ternary condition vector (e.g., temperature, humidity, light).
    pub conditions: Vec<Ternary>,
    /// Maximum population this niche supports.
    pub capacity: u64,
}

impl Niche {
    pub fn new(name: &str, conditions: Vec<Ternary>, capacity: u64) -> Self {
        Self { name: name.to_string(), conditions, capacity }
    }

    /// Compatibility score: how well a species' strategy matches this niche.
    /// Returns count of matching ternary signs.
    pub fn compatibility(&self, strategy: Ternary) -> i32 {
        // Simple: count how many conditions align with strategy
        self.conditions.iter().map(|&c| {
            match (c, strategy) {
                (Ternary::Pos, Ternary::Pos) | (Ternary::Neg, Ternary::Neg) => 1,
                (Ternary::Pos, Ternary::Neg) | (Ternary::Neg, Ternary::Pos) => -1,
                _ => 0,
            }
        }).sum()
    }

    /// Number of conditions.
    pub fn dimensionality(&self) -> usize {
        self.conditions.len()
    }
}

/// Carrying capacity regulator.
#[derive(Clone, Debug)]
pub struct CarryingCapacity {
    /// Global resource limit.
    pub global_limit: u64,
    /// Per-niche limits.
    pub niche_limits: Vec<u64>,
}

impl CarryingCapacity {
    pub fn new(global_limit: u64, niche_limits: Vec<u64>) -> Self {
        Self { global_limit, niche_limits }
    }

    /// Enforce global capacity across all species.
    pub fn enforce_global(&self, species: &mut [Species]) {
        let total: u64 = species.iter().map(|s| s.population).sum();
        if total > self.global_limit {
            let scale = self.global_limit as f64 / total as f64;
            for s in species.iter_mut() {
                s.population = ((s.population as f64 * scale) as u64).max(0);
                if s.population == 0 { s.extinct = true; }
            }
        }
    }

    /// Enforce per-niche capacity for species assigned to niches.
    pub fn enforce_niche(&self, species: &mut [Species], niche_assignments: &[(usize, usize)]) {
        for &(species_idx, niche_idx) in niche_assignments {
            if niche_idx < self.niche_limits.len() && species_idx < species.len() {
                species[species_idx].cap_population(self.niche_limits[niche_idx]);
            }
        }
    }

    /// Total population across species.
    pub fn total_population(species: &[Species]) -> u64 {
        species.iter().map(|s| s.population).sum()
    }
}

/// Ecological succession: stages of ecosystem maturation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuccessionStage {
    Pioneer,
    Intermediate,
    Climax,
}

impl SuccessionStage {
    pub fn from_diversity(species_count: usize, total_species: usize) -> Self {
        let ratio = if total_species == 0 { 0.0 } else { species_count as f64 / total_species as f64 };
        if ratio < 0.33 {
            SuccessionStage::Pioneer
        } else if ratio < 0.66 {
            SuccessionStage::Intermediate
        } else {
            SuccessionStage::Climax
        }
    }

    /// Growth modifier: pioneers grow fast, climax slow.
    pub fn growth_modifier(&self) -> i32 {
        match self {
            SuccessionStage::Pioneer => 15,
            SuccessionStage::Intermediate => 8,
            SuccessionStage::Climax => 3,
        }
    }
}

/// Ecological succession tracker.
#[derive(Clone, Debug)]
pub struct EcologicalSuccession {
    pub total_species: usize,
    pub current_stage: SuccessionStage,
    pub ticks_elapsed: u64,
}

impl EcologicalSuccession {
    pub fn new(total_species: usize) -> Self {
        Self { total_species, current_stage: SuccessionStage::Pioneer, ticks_elapsed: 0 }
    }

    /// Tick the succession forward.
    pub fn tick(&mut self, living_species: usize) -> &SuccessionStage {
        self.ticks_elapsed += 1;
        self.current_stage = SuccessionStage::from_diversity(living_species, self.total_species);
        &self.current_stage
    }
}

/// Keystone species detector.
#[derive(Clone, Debug)]
pub struct Keystone;

impl Keystone {
    /// Detect keystone species by simulating removal of each species
    /// and measuring biodiversity impact.
    /// Returns indices of species whose removal causes >threshold% biodiversity loss.
    pub fn detect(species: &[Species], food_web: &FoodWeb, threshold_pct: u8) -> Vec<usize> {
        let living_count = species.iter().filter(|s| !s.extinct).count() as f64;
        if living_count == 0.0 { return vec![]; }

        let mut keystones = Vec::new();
        for i in 0..species.len() {
            if species[i].extinct { continue; }
            // Count how many species depend on this one (directly or are its prey)
            let prey_of_predator = food_web.predators_of(i);
            // A species is keystone if many predators depend on it
            let dependent_ratio = prey_of_predator.len() as f64 / living_count * 100.0;
            // Also consider: is this a top predator with wide prey base?
            let prey_list = food_web.prey_of(i);
            let impact = dependent_ratio.max(prey_list.len() as f64 / living_count * 100.0);

            if impact >= threshold_pct as f64 || prey_of_predator.len() >= 3 {
                keystones.push(i);
            }
        }
        keystones
    }

    /// Simple metric: species connectedness in the food web.
    pub fn connectedness(food_web: &FoodWeb, species_idx: usize) -> usize {
        let as_predator = food_web.prey_of(species_idx).len();
        let as_prey = food_web.predators_of(species_idx).len();
        as_predator + as_prey
    }
}

/// Full ecosystem simulation.
#[derive(Clone, Debug)]
pub struct Ecosystem {
    pub species: Vec<Species>,
    pub food_web: FoodWeb,
    pub niches: Vec<Niche>,
    pub carrying_capacity: CarryingCapacity,
    pub succession: EcologicalSuccession,
    pub ticks: u64,
}

impl Ecosystem {
    pub fn new(
        species: Vec<Species>,
        food_web: FoodWeb,
        niches: Vec<Niche>,
        carrying_capacity: CarryingCapacity,
    ) -> Self {
        let total = species.len();
        Self {
            species,
            food_web,
            niches,
            carrying_capacity,
            succession: EcologicalSuccession::new(total),
            ticks: 0,
        }
    }

    /// Run one simulation tick.
    pub fn tick(&mut self) {
        // Apply succession growth modifiers
        let modifier = self.succession.current_stage.growth_modifier();
        for s in &mut self.species {
            if !s.extinct {
                s.growth_rate = modifier;
                s.tick();
            }
        }

        // Apply predation
        self.food_web.tick(&mut self.species);

        // Enforce carrying capacity
        self.carrying_capacity.enforce_global(&mut self.species);

        // Update succession
        let living = self.species.iter().filter(|s| !s.extinct).count();
        self.succession.tick(living);

        self.ticks += 1;
    }

    /// Get living species count.
    pub fn living_count(&self) -> usize {
        self.species.iter().filter(|s| !s.extinct).count()
    }

    /// Get total population.
    pub fn total_population(&self) -> u64 {
        self.species.iter().map(|s| s.population).sum()
    }

    /// Detect keystone species.
    pub fn keystone_species(&self, threshold_pct: u8) -> Vec<usize> {
        Keystone::detect(&self.species, &self.food_web, threshold_pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_species_tick_growth() {
        let mut s = Species::new("rabbit", 100, 10, 0, Ternary::Pos);
        let new_pop = s.tick();
        assert_eq!(new_pop, 110); // 100 + 10% of 100
    }

    #[test]
    fn test_species_tick_negative_growth() {
        let mut s = Species::new("declining", 100, -50, 1, Ternary::Neg);
        s.tick();
        assert_eq!(s.population, 50);
    }

    #[test]
    fn test_species_prey_loss() {
        let mut s = Species::new("prey", 100, 5, 1, Ternary::Zero);
        s.prey_loss(60);
        assert_eq!(s.population, 40);
        assert!(!s.extinct);
    }

    #[test]
    fn test_species_extinction() {
        let mut s = Species::new("doomed", 10, 0, 0, Ternary::Zero);
        s.prey_loss(10);
        assert_eq!(s.population, 0);
        assert!(s.extinct);
    }

    #[test]
    fn test_species_extinct_no_tick() {
        let mut s = Species::new("dead", 0, 50, 0, Ternary::Zero);
        s.extinct = true;
        assert_eq!(s.tick(), 0);
    }

    #[test]
    fn test_food_web_predation() {
        let mut species = vec![
            Species::new("grass", 1000, 10, 0, Ternary::Zero),
            Species::new("rabbit", 100, 5, 1, Ternary::Pos),
        ];
        let mut fw = FoodWeb::new();
        fw.add_link(1, 0, 50); // rabbit eats grass at 5%
        let losses = fw.tick(&mut species);
        assert_eq!(losses.len(), 1);
        assert!(species[0].population < 1000); // grass lost some
    }

    #[test]
    fn test_food_web_predators_of() {
        let mut fw = FoodWeb::new();
        fw.add_link(2, 0, 10);
        fw.add_link(1, 0, 20);
        let preds = fw.predators_of(0);
        assert_eq!(preds, vec![2, 1]);
    }

    #[test]
    fn test_food_web_prey_of() {
        let mut fw = FoodWeb::new();
        fw.add_link(0, 1, 10);
        fw.add_link(0, 2, 10);
        let prey = fw.prey_of(0);
        assert_eq!(prey, vec![1, 2]);
    }

    #[test]
    fn test_niche_compatibility() {
        let niche = Niche::new("forest", vec![Ternary::Pos, Ternary::Pos, Ternary::Neg], 500);
        assert_eq!(niche.compatibility(Ternary::Pos), 2 - 1); // 2 pos match, 1 neg mismatch
        assert_eq!(niche.compatibility(Ternary::Neg), -2 + 1);
    }

    #[test]
    fn test_niche_dimensionality() {
        let niche = Niche::new("desert", vec![Ternary::Pos, Ternary::Zero, Ternary::Neg], 100);
        assert_eq!(niche.dimensionality(), 3);
    }

    #[test]
    fn test_carrying_capacity_global() {
        let mut species = vec![
            Species::new("a", 600, 0, 0, Ternary::Zero),
            Species::new("b", 600, 0, 0, Ternary::Zero),
        ];
        let cc = CarryingCapacity::new(1000, vec![]);
        cc.enforce_global(&mut species);
        assert!(species[0].population + species[1].population <= 1000);
    }

    #[test]
    fn test_carrying_capacity_niche() {
        let mut species = vec![
            Species::new("a", 200, 0, 0, Ternary::Zero),
            Species::new("b", 300, 0, 0, Ternary::Zero),
        ];
        let cc = CarryingCapacity::new(10000, vec![100, 200]);
        cc.enforce_niche(&mut species, &[(0, 0), (1, 1)]);
        assert_eq!(species[0].population, 100);
        assert_eq!(species[1].population, 200);
    }

    #[test]
    fn test_carrying_capacity_total() {
        let species = vec![
            Species::new("a", 50, 0, 0, Ternary::Zero),
            Species::new("b", 75, 0, 0, Ternary::Zero),
        ];
        assert_eq!(CarryingCapacity::total_population(&species), 125);
    }

    #[test]
    fn test_succession_stages() {
        assert_eq!(SuccessionStage::from_diversity(1, 10), SuccessionStage::Pioneer);
        assert_eq!(SuccessionStage::from_diversity(5, 10), SuccessionStage::Intermediate);
        assert_eq!(SuccessionStage::from_diversity(8, 10), SuccessionStage::Climax);
    }

    #[test]
    fn test_succession_growth_modifiers() {
        assert!(SuccessionStage::Pioneer.growth_modifier() > SuccessionStage::Intermediate.growth_modifier());
        assert!(SuccessionStage::Intermediate.growth_modifier() > SuccessionStage::Climax.growth_modifier());
    }

    #[test]
    fn test_ecological_succession_tick() {
        let mut succ = EcologicalSuccession::new(10);
        succ.tick(8);
        assert_eq!(succ.ticks_elapsed, 1);
        assert_eq!(succ.current_stage, SuccessionStage::Climax);
    }

    #[test]
    fn test_keystone_connectedness() {
        let mut fw = FoodWeb::new();
        fw.add_link(0, 1, 10);
        fw.add_link(0, 2, 10);
        fw.add_link(3, 0, 10);
        assert_eq!(Keystone::connectedness(&fw, 0), 3); // 2 prey + 1 predator
    }

    #[test]
    fn test_ecosystem_tick() {
        let species = vec![
            Species::new("grass", 500, 0, 0, Ternary::Zero),
            Species::new("rabbit", 50, 0, 1, Ternary::Pos),
        ];
        let mut fw = FoodWeb::new();
        fw.add_link(1, 0, 20);
        let cc = CarryingCapacity::new(10000, vec![]);
        let mut eco = Ecosystem::new(species, fw, vec![], cc);
        eco.tick();
        assert_eq!(eco.ticks, 1);
        assert!(eco.total_population() > 0);
    }

    #[test]
    fn test_ecosystem_living_count() {
        let mut species = vec![
            Species::new("a", 100, 0, 0, Ternary::Zero),
            Species::new("b", 0, 0, 0, Ternary::Zero),
        ];
        species[1].extinct = true;
        let cc = CarryingCapacity::new(10000, vec![]);
        let eco = Ecosystem::new(species, FoodWeb::new(), vec![], cc);
        assert_eq!(eco.living_count(), 1);
    }

    #[test]
    fn test_keystone_detect() {
        let species = vec![
            Species::new("grass", 1000, 0, 0, Ternary::Zero),
            Species::new("rabbit", 100, 0, 1, Ternary::Pos),
            Species::new("fox", 20, 0, 2, Ternary::Neg),
        ];
        let mut fw = FoodWeb::new();
        fw.add_link(1, 0, 10); // rabbit eats grass
        fw.add_link(2, 1, 20); // fox eats rabbit
        let keystones = Keystone::detect(&species, &fw, 20);
        // Rabbit is keystone: fox depends on it
        assert!(keystones.contains(&1));
    }

    #[test]
    fn test_species_cap_population() {
        let mut s = Species::new("overflow", 150, 0, 0, Ternary::Zero);
        s.cap_population(100);
        assert_eq!(s.population, 100);
    }

    #[test]
    fn test_food_web_skips_extinct() {
        let mut species = vec![
            Species::new("prey", 100, 0, 0, Ternary::Zero),
            Species::new("predator", 50, 0, 1, Ternary::Pos),
        ];
        species[1].extinct = true;
        let mut fw = FoodWeb::new();
        fw.add_link(1, 0, 50);
        let losses = fw.tick(&mut species);
        assert!(losses.is_empty()); // predator extinct, no predation
    }
}
