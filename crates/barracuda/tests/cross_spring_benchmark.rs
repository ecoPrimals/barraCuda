// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-Spring Evolution Benchmark Harness
//!
//! Benchmarks the modern absorption primitives that flow across spring
//! domains: Welford statistics, tolerance system, Verlet neighbor list,
//! eps precision guards, and the provenance registry itself.
//!
//! Records wall time and throughput for each primitive so we can track
//! how cross-spring evolution affects performance over time.
//!
//! Run: `cargo test --test cross_spring_benchmark -- --nocapture`

use std::time::Instant;

#[derive(Debug)]
struct BenchResult {
    name: &'static str,
    origin: &'static str,
    consumers: &'static str,
    #[expect(dead_code, reason = "used by Debug derive in --nocapture output")]
    n: usize,
    wall_us: f64,
    throughput: String,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<40} {:>8.1} µs  {:>12}  ({} → {})",
            self.name, self.wall_us, self.throughput, self.origin, self.consumers
        )
    }
}

fn bench_welford_univariate(n: usize) -> BenchResult {
    use barracuda::stats::welford::WelfordState;

    let data: Vec<f64> = (0..n).map(|i| 1e9 + (i as f64) * 0.001).collect();

    let start = Instant::now();
    let state = WelfordState::from_slice(&data);
    let elapsed = start.elapsed();

    assert!(state.count() == n as u64);
    assert!(state.mean().is_finite());

    BenchResult {
        name: "Welford univariate (mean + variance)",
        origin: "groundSpring V80",
        consumers: "all springs",
        n,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!("{:.1}M pts/s", n as f64 / elapsed.as_secs_f64() / 1e6),
    }
}

fn bench_welford_covariance(n: usize) -> BenchResult {
    use barracuda::stats::welford::WelfordCovState;

    let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + x.sin() * 0.01).collect();

    let start = Instant::now();
    let state = WelfordCovState::from_slices(&xs, &ys);
    let elapsed = start.elapsed();

    assert!(state.correlation().abs() > 0.99);

    BenchResult {
        name: "Welford bivariate (covariance + corr)",
        origin: "groundSpring V80",
        consumers: "hotSpring, neuralSpring",
        n,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!("{:.1}M pairs/s", n as f64 / elapsed.as_secs_f64() / 1e6),
    }
}

fn bench_welford_parallel_merge(n: usize, chunks: usize) -> BenchResult {
    use barracuda::stats::welford::WelfordState;

    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
    let chunk_size = n / chunks;

    let start = Instant::now();
    let mut states: Vec<WelfordState> = data
        .chunks(chunk_size)
        .map(WelfordState::from_slice)
        .collect();

    let mut final_state = states.remove(0);
    for s in &states {
        final_state.merge(s);
    }
    let elapsed = start.elapsed();

    assert!(final_state.count() == n as u64);

    BenchResult {
        name: "Welford parallel merge (Chan's algorithm)",
        origin: "groundSpring V80",
        consumers: "all springs (GPU reduce pattern)",
        n,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!(
            "{:.1}M pts/s ({chunks} chunks)",
            n as f64 / elapsed.as_secs_f64() / 1e6
        ),
    }
}

fn bench_tolerance_comparisons(n: usize) -> BenchResult {
    use barracuda::numerical::tolerance::Tolerance;

    let pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let a = (i as f64) * 0.001;
            let b = a + 1e-13;
            (a, b)
        })
        .collect();

    let tiers = [
        Tolerance::CPU_F64,
        Tolerance::GPU_F64,
        Tolerance::DF64,
        Tolerance::F32,
    ];

    let start = Instant::now();
    let mut pass_count = 0usize;
    for &(a, b) in &pairs {
        for &tier in &tiers {
            if tier.approx_eq(a, b) {
                pass_count += 1;
            }
        }
    }
    let elapsed = start.elapsed();

    let total_checks = n * tiers.len();
    assert!(pass_count > 0);

    BenchResult {
        name: "Tolerance 4-tier comparison sweep",
        origin: "groundSpring V76",
        consumers: "all springs (validation)",
        n: total_checks,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!(
            "{:.1}M cmp/s",
            total_checks as f64 / elapsed.as_secs_f64() / 1e6
        ),
    }
}

fn bench_verlet_list_build(n_particles: usize) -> BenchResult {
    use barracuda::ops::md::neighbor::VerletList;

    let box_side = (n_particles as f64).cbrt() * 2.0;
    let rc = 2.5;
    let r_skin = 0.3;

    let mut rng = 42u64;
    let positions: Vec<f64> = (0..n_particles * 3)
        .map(|_| {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (rng as f64 / u64::MAX as f64) * box_side
        })
        .collect();

    let mut vl = VerletList::new(rc, r_skin, box_side);

    let start = Instant::now();
    vl.build(&positions, n_particles);
    let elapsed = start.elapsed();

    assert!(vl.total_pairs() > 0);

    BenchResult {
        name: "Verlet neighbor list build (CSR)",
        origin: "hotSpring MD",
        consumers: "hotSpring, wetSpring (bio-materials)",
        n: n_particles,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!(
            "{:.1}K particles/ms, {} pairs",
            n_particles as f64 / elapsed.as_secs_f64() / 1e3,
            vl.total_pairs()
        ),
    }
}

fn bench_verlet_needs_rebuild(n_particles: usize) -> BenchResult {
    use barracuda::ops::md::neighbor::VerletList;

    let box_side = (n_particles as f64).cbrt() * 2.0;
    let rc = 2.5;
    let r_skin = 0.3;

    let mut rng = 42u64;
    let positions: Vec<f64> = (0..n_particles * 3)
        .map(|_| {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (rng as f64 / u64::MAX as f64) * box_side
        })
        .collect();

    let mut vl = VerletList::new(rc, r_skin, box_side);
    vl.build(&positions, n_particles);

    let iters = 10_000;
    let start = Instant::now();
    let mut rebuild = false;
    for _ in 0..iters {
        rebuild |= vl.needs_rebuild(&positions);
    }
    let elapsed = start.elapsed();

    assert!(!rebuild, "no displacement means no rebuild");

    BenchResult {
        name: "Verlet needs_rebuild check (amortized)",
        origin: "hotSpring MD",
        consumers: "all MD springs",
        n: iters,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!(
            "{:.0} ns/check ({n_particles} particles)",
            elapsed.as_nanos() as f64 / iters as f64
        ),
    }
}

fn bench_eps_guard_application(n: usize) -> BenchResult {
    use barracuda::shaders::precision::eps;

    let values: Vec<f64> = (0..n)
        .map(|i| {
            if i % 100 == 0 {
                0.0
            } else {
                (i as f64) * 1e-200
            }
        })
        .collect();

    let start = Instant::now();
    let mut safe_sum = 0.0f64;
    for &v in &values {
        safe_sum += 1.0 / v.max(eps::SAFE_DIV);
    }
    let elapsed = start.elapsed();

    assert!(safe_sum.is_finite());

    BenchResult {
        name: "eps::SAFE_DIV guard application",
        origin: "groundSpring V76",
        consumers: "all springs (GPU shader preamble)",
        n,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!("{:.1}M ops/s", n as f64 / elapsed.as_secs_f64() / 1e6),
    }
}

fn bench_provenance_registry_queries() -> BenchResult {
    use barracuda::shaders::provenance::{self, SpringDomain};

    let iters = 10_000;
    let start = Instant::now();
    let mut total = 0usize;
    for _ in 0..iters {
        total += provenance::cross_spring_shaders().len();
        total += provenance::shaders_from(SpringDomain::HOT_SPRING).len();
        total += provenance::shaders_consumed_by(SpringDomain::WET_SPRING).len();
        let _ = provenance::cross_spring_matrix();
    }
    let elapsed = start.elapsed();

    assert!(total > 0);

    BenchResult {
        name: "Provenance registry query suite",
        origin: "barraCuda",
        consumers: "cross-spring auditing",
        n: iters * 4,
        wall_us: elapsed.as_secs_f64() * 1e6,
        throughput: format!(
            "{:.0} ns/query",
            elapsed.as_nanos() as f64 / (iters as f64 * 4.0)
        ),
    }
}

#[test]
fn cross_spring_evolution_benchmark() {
    let n = 100_000;

    let results = vec![
        bench_welford_univariate(n),
        bench_welford_covariance(n),
        bench_welford_parallel_merge(n, 8),
        bench_tolerance_comparisons(n),
        bench_verlet_list_build(2000),
        bench_verlet_needs_rebuild(2000),
        bench_eps_guard_application(n),
        bench_provenance_registry_queries(),
    ];

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Cross-Spring Evolution Benchmark — barraCuda v0.3.3               ║");
    println!("║          hotSpring precision · wetSpring bio · neuralSpring ML              ║");
    println!("║          groundSpring tolerance · airSpring hydrology                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    for r in &results {
        println!("║ {r}");
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    println!("Cross-Spring Shader Flow:");
    let matrix = barracuda::shaders::provenance::cross_spring_matrix();
    let mut flows: Vec<_> = matrix.iter().collect();
    flows.sort_by(|a, b| b.1.cmp(a.1));
    for ((from, to), count) in &flows {
        println!("  {from} → {to}: {count} shared shaders");
    }
    println!();
}
