// SPDX-License-Identifier: AGPL-3.0-or-later
//! Echo State Network Time Series Prediction Demo
//!
//! This example demonstrates using the high-level ESN API for time series
//! prediction. We train an ESN to learn a simple sine wave pattern and then
//! use it to predict future values.
//!
//! Run with: `cargo test --package barracuda --example esn_demo`

use barracuda::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Echo State Network - Time Series Prediction Demo\n");
    println!("═══════════════════════════════════════════════════\n");

    // No GPU needed - pure Rust!
    println!("✅ Running on CPU (pure Rust!)\n");

    // Configure ESN
    println!("⚙️  Configuring ESN...");
    let config = ESNConfig {
        input_size: 1,
        reservoir_size: 100,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.1,
        leak_rate: 0.3,
        regularization: 1e-6,
        seed: 42,
    };

    println!("   Reservoir: {} neurons", config.reservoir_size);
    println!("   Spectral radius: {:.2}", config.spectral_radius);
    println!();

    // Create ESN (pure Rust - no device needed!)
    println!("🧠 Creating ESN...");
    let mut esn = ESN::new(config).await?;
    println!("✅ ESN initialized (pure Rust!)\n");

    // Generate training data
    println!("📊 Generating training data (sine wave)...");
    let num_train = 100;
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();

    for i in 0..num_train {
        let t = i as f32 * 0.1;
        train_inputs.push(vec![(t).sin()]);
        train_targets.push(vec![(t + 0.1).sin()]);
    }

    println!("   {} training samples\n", num_train);

    // Train (pure Rust - no GPU!)
    println!("🎓 Training ESN...");
    let mse = esn.train(&train_inputs, &train_targets).await?;
    println!("✅ Training MSE: {:.6}\n", mse);

    // Test
    println!("🔮 Testing predictions...");
    esn.reset_state().await?;
    let test_inputs: Vec<Vec<f32>> = (0..20)
        .map(|i| vec![((num_train + i) as f32 * 0.1).sin()])
        .collect();

    // Predict for each input individually
    let mut predictions = Vec::new();
    for input in &test_inputs {
        let pred = esn.predict(input).await?;
        predictions.push(pred);
    }

    println!(
        "✅ {} predictions generated (pure Rust!)\n",
        predictions.len()
    );

    println!("🎊 Demo Complete!");
    println!("\n💡 Key Achievement:");
    println!("   Universal compute: Same code runs on NPU/GPU/CPU!");

    Ok(())
}
