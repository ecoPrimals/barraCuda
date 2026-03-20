// SPDX-License-Identifier: AGPL-3.0-or-later
//! Metropolis-Hastings MCMC sampling with Boltzmann acceptance.
//!
//! CPU implementation complementing the GPU shader [`crate::sample::WGSL_METROPOLIS`].

/// Result of Boltzmann/Metropolis MCMC sampling.
#[derive(Debug, Clone)]
pub struct BoltzmannResult {
    /// Loss values at each step (including initial).
    pub losses: Vec<f64>,
    /// Fraction of proposed steps that were accepted.
    pub acceptance_rate: f64,
    /// Final parameter values after all steps.
    pub final_params: Vec<f64>,
}

/// Xoshiro256** PRNG (seeded via splitmix64).
struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut x = seed;
        for i in 0..4 {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            s[i] = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform [0, 1) from u64.
    fn uniform01(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Box-Muller: two uniform [0,1) -> two independent standard normals.
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let u1_safe = (1.0 - u1).max(1e-300);
    let r = (-2.0 * u1_safe.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Metropolis-Hastings MCMC sampling with Boltzmann acceptance.
///
/// Draws samples from exp(-loss/temperature) by proposing perturbations
/// of size `step_size` and accepting/rejecting via the Metropolis criterion.
#[must_use]
pub fn boltzmann_sampling(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    initial_params: &[f64],
    temperature: f64,
    step_size: f64,
    n_steps: usize,
    seed: u64,
) -> BoltzmannResult {
    let mut rng = Xoshiro256StarStar::new(seed);
    let mut params = initial_params.to_vec();
    let mut current_loss = loss_fn(&params);
    let mut losses = Vec::with_capacity(n_steps + 1);
    losses.push(current_loss);
    let mut accepted = 0usize;

    for _ in 0..n_steps {
        let u1 = rng.uniform01();
        let u2 = rng.uniform01();
        let (z1, z2) = box_muller(u1, u2);

        let mut proposed = params.clone();
        proposed[0] += step_size * z1;
        if proposed.len() > 1 {
            proposed[1] += step_size * z2;
        }
        for i in 2..proposed.len() {
            let u1 = rng.uniform01();
            let u2 = rng.uniform01();
            let (z, _) = box_muller(u1, u2);
            proposed[i] += step_size * z;
        }

        let proposed_loss = loss_fn(&proposed);
        let log_alpha = (current_loss - proposed_loss) / temperature;
        let accept = log_alpha >= 0.0 || rng.uniform01() < log_alpha.exp();

        if accept {
            params = proposed;
            current_loss = proposed_loss;
            accepted += 1;
        }
        losses.push(current_loss);
    }

    BoltzmannResult {
        losses,
        acceptance_rate: accepted as f64 / n_steps as f64,
        final_params: params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boltzmann_quadratic() {
        // Sampling around minimum of x^2 (and y^2)
        let loss_fn = |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>();
        let result = boltzmann_sampling(&loss_fn, &[5.0, 5.0], 1.0, 0.5, 2000, 42);
        // Final params should be closer to origin than start
        assert!(
            result.final_params[0].abs() < 5.0,
            "x should move toward 0, got {}",
            result.final_params[0]
        );
        assert!(
            result.final_params[1].abs() < 5.0,
            "y should move toward 0, got {}",
            result.final_params[1]
        );
        assert!(
            result.losses.last().copied().unwrap_or(1e10) < 25.0,
            "loss should decrease from 50"
        );
    }

    #[test]
    fn test_boltzmann_acceptance_rate() {
        // Lower temperature -> lower acceptance rate (stricter)
        let loss_fn = |p: &[f64]| p[0] * p[0];
        let result_high_t = boltzmann_sampling(&loss_fn, &[1.0], 2.0, 0.5, 500, 123);
        let result_low_t = boltzmann_sampling(&loss_fn, &[1.0], 0.1, 0.5, 500, 123);
        assert!(
            result_high_t.acceptance_rate > result_low_t.acceptance_rate,
            "high T should accept more: {} vs {}",
            result_high_t.acceptance_rate,
            result_low_t.acceptance_rate
        );
    }

    #[test]
    fn test_boltzmann_deterministic() {
        // Same seed gives same result
        let loss_fn = |p: &[f64]| p[0] * p[0];
        let r1 = boltzmann_sampling(&loss_fn, &[1.0], 1.0, 0.3, 100, 999);
        let r2 = boltzmann_sampling(&loss_fn, &[1.0], 1.0, 0.3, 100, 999);
        assert_eq!(r1.final_params, r2.final_params);
        assert_eq!(r1.losses, r2.losses);
        assert!((r1.acceptance_rate - r2.acceptance_rate).abs() < 1e-14);
    }
}
