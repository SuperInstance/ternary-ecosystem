# ternary-ecosystem — Full ecosystem simulation with multiple ternary species

Ecosystem struct, species definitions, food webs, niches, carrying capacity, ecological succession tracking, and keystone species detection for ternary agent populations.

## Why This Exists

Individual ternary agents don't exist in isolation — they form populations, compete for resources, and prey on each other. Simulating ecology requires more than just agent-level logic. This crate provides the population-level structures: who eats whom, how many the environment can support, which species are keystone, and how the ecosystem matures over time.

## Core Concepts

- **Balanced ternary** — Values -1, 0, +1. In this crate, a species' strategy trait is ternary: Neg means conservative, Zero means neutral, Pos means aggressive.
- **Species** — An agent type with population, growth rate, trophic level, and a ternary strategy trait. Species go extinct when population hits zero.
- **FoodWeb** — A directed graph of predation links (predator → prey, with a rate). Each tick, predators consume prey proportional to predator population and predation rate.
- **Niche** — An environmental space described by a ternary condition vector (e.g., [hot, dry, bright]). Species' strategy compatibility with niche conditions determines fitness.
- **CarryingCapacity** — Enforces population limits. Global capacity applies across all species (proportional scaling when exceeded). Per-niche capacity caps species assigned to specific niches.
- **EcologicalSuccession** — Tracks ecosystem maturity through three stages: Pioneer (low diversity, fast growth), Intermediate, and Climax (high diversity, slow growth). Growth rates adapt to the current stage.
- **Keystone** — A species whose removal would cause disproportionate biodiversity loss. Detected by analyzing food web connectivity: species with many dependents or wide prey bases.

## Quick Start

```toml
[dependencies]
ternary-ecosystem = "0.1"
```

```rust
use ternary_ecosystem::*;

// Define species
let species = vec![
    Species::new("grass",   1000, 10, 0, Ternary::Zero),
    Species::new("rabbit",   100,  5, 1, Ternary::Pos),
    Species::new("fox",       20,  3, 2, Ternary::Neg),
];

// Build food web: rabbit eats grass, fox eats rabbit
let mut food_web = FoodWeb::new();
food_web.add_link(1, 0, 30); // rabbit → grass, 3% rate
food_web.add_link(2, 1, 50); // fox → rabbit, 5% rate

// Set up ecosystem
let cc = CarryingCapacity::new(5000, vec![2000, 500, 100]);
let mut eco = Ecosystem::new(species, food_web, vec![], cc);

// Run simulation
for _ in 0..100 {
    eco.tick();
}
println!("Living species: {}", eco.living_count());
println!("Total population: {}", eco.total_population());

// Find keystone species
let keystones = eco.keystone_species(20);
```

## API Overview

| Type | Purpose |
|------|---------|
| `Species` | Agent type with population, growth, trophic level, ternary strategy |
| `FoodWeb` | Predator-prey relationship network with predation rates |
| `Niche` | Environmental space with ternary conditions and capacity |
| `CarryingCapacity` | Global and per-niche population limits |
| `EcologicalSuccession` | Tracks Pioneer → Intermediate → Climax maturation |
| `Keystone` | Detects species whose removal collapses diversity |
| `Ecosystem` | Full simulation: species + food web + capacity + succession |

## How It Works

Each tick: species grow (rate × population / 100), predation reduces prey populations, carrying capacity caps totals, and succession stage updates. Growth rates are overridden by the succession stage modifier — pioneers grow fast (15%), climax species grow slowly (3%). This creates natural deceleration as the ecosystem matures.

Keystone detection works by analyzing food web topology: species that many predators depend on, or that have wide prey bases, are flagged as keystone. The threshold is configurable.

## Known Limitations

- **Lotka-Volterra is simplified** — Predation is a fixed-rate linear model, not the coupled differential equations. No oscillatory dynamics emerge naturally.
- **No spatial structure** — All species share the same space. No territory, migration, or distance-based interactions.
- **Growth rate is overwritten** — The succession stage modifier replaces species' individual growth rates each tick. Custom growth dynamics are lost.
- **No stochastic extinction** — Extinction only happens when population hits exactly zero. No chance-based extinction at low populations.
- **Niche compatibility is simplistic** — Just counts sign matches between niche conditions and strategy. No multi-dimensional fitness landscape.

## Use Cases

1. **Room population dynamics** — Simulate how ternary agent populations in a room grow, compete, and stabilize over time, with predator-prey dynamics between agent types.
2. **Strategy ecology** — Model how aggressive (Pos), neutral (Zero), and conservative (Neg) strategies compete, finding stable equilibria.
3. **Keystone identification** — Find which agent types are critical to ecosystem health — removing them causes cascade extinction.

## Ecosystem Context

Part of the SuperInstance ternary crate family. `ternary-ecosystem` uses `ternary-genome` for evolving species traits and feeds into `ternary-room` for room-level simulation. Combines concepts from `ternary-game-theory` (strategy interactions) and population dynamics (Lotka-Volterra inspired).

## License

MIT

## See Also
- **ternary-cell** — related
- **ternary-genome** — related
- **ternary-fitness** — related
- **ternary-evolution-advanced** — related
- **ternary-room** — related

