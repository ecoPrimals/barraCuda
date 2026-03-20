// SPDX-License-Identifier: AGPL-3.0-or-later
//! Statistical functions
//!
//! Provides core statistical distributions, measures, and agreement metrics
//! for scientific computing across all springs.
//!
//! # Agreement Metrics (S64 absorption)
//!
//! - **RMSE, MAE, MBE, NSE, R², IA, `hit_rate`**: Model validation
//! - **mean, percentile**: Descriptive statistics
//! - **dot, `l2_norm`**: CPU vector operations
//!
//! # Ecological Diversity (S64 absorption)
//!
//! - **Shannon, Simpson, Chao1, Pielou**: Alpha diversity
//! - **Bray-Curtis**: Beta diversity (pairwise dissimilarity)
//! - **Rarefaction curves**: Expected species vs subsampling depth
//!
//! # Distributions
//!
//! - **Normal (Gaussian)**: CDF, PDF, inverse CDF (probit/quantile)
//!
//! # Correlation & Covariance
//!
//! - **Pearson/Spearman correlation**, **Covariance**, **Correlation matrix**
//!
//! # Chi-Squared / Bootstrap
//!
//! - **`chi2_decomposed`**: Per-datum residuals, pulls, contributions
//! - **`bootstrap_ci`**: Non-parametric confidence intervals
//!
//! # Regression (S66 absorption from airSpring)
//!
//! - **Linear, Quadratic, Exponential, Logarithmic**: Closed-form least-squares
//!
//! # Hydrology (S66+S70+S81 absorption from airSpring/groundSpring)
//!
//! - **FAO-56 Penman-Monteith ET₀** (scalar, full equation)
//! - **Hargreaves, Thornthwaite, Makkink, Turc, Hamon ET₀**: Tier A GPU-ready primitives
//! - **Crop coefficient, Soil water balance**: FAO-56 reference
//!
//! # Population Genetics (S70 absorption from groundSpring)
//!
//! - **Kimura fixation probability**, **Error threshold** (quasispecies)
//! - **Detection power/threshold** for rare taxa
//!
//! # Jackknife (S70 absorption from groundSpring)
//!
//! - **Leave-one-out jackknife** for mean and arbitrary statistics
//!
//! # Moving Window f64 (S66 absorption from airSpring)
//!
//! - **CPU f64 sliding window**: mean, variance, min, max
//!
//! # References
//!
//! - Abramowitz & Stegun §26, Moro (1995), Efron & Tibshirani (1993)
//! - QIIME2/skbio for diversity metrics, Willmott (1981) for IA
//! - Hargreaves & Samani (1985), FAO-56 (Allen et al. 1998)
//! - Dong et al. (2020) *Agriculture* 10(12):598

/// WGSL kernel for GPU-parallel bootstrap mean estimation (f64).
#[cfg(feature = "gpu")]
pub const WGSL_BOOTSTRAP_MEAN_F64: &str =
    include_str!("../shaders/special/bootstrap_mean_f64.wgsl");

/// WGSL kernel for parallel histogram via atomic binning (f64 canonical).
#[cfg(feature = "gpu")]
pub const WGSL_HISTOGRAM_F64: &str = include_str!("../shaders/stats/histogram_f64.wgsl");

/// WGSL kernel for parallel histogram (f32 downcast for devices without f64).
#[cfg(feature = "gpu")]
pub static WGSL_HISTOGRAM_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| include_str!("../shaders/stats/histogram_f64.wgsl").to_string());

pub mod bootstrap;
pub mod chi2;
pub mod correlation;
pub mod diversity;
pub mod evolution;
pub mod histogram;
pub mod hydrology;
pub mod jackknife;
pub mod metrics;
pub mod moving_window_f64;
pub mod normal;
pub mod regression;
pub mod spectral_density;
pub mod welford;

#[cfg(feature = "gpu")]
pub use bootstrap::BootstrapMeanGpu;
pub use bootstrap::{
    BootstrapCI, bootstrap_ci, bootstrap_mean, bootstrap_median, bootstrap_std, rawr_mean,
};
pub use chi2::{Chi2Decomposed, chi2_decomposed, chi2_decomposed_weighted};
pub use correlation::{
    correlation_matrix, covariance, covariance_matrix, pearson_correlation, spearman_correlation,
};
pub use diversity::{
    AlphaDiversity, alpha_diversity, bray_curtis, bray_curtis_condensed, bray_curtis_matrix, chao1,
    chao1_classic, condensed_index, observed_features, pielou_evenness, rarefaction_curve, shannon,
    shannon_from_frequencies, simpson,
};
#[cfg(feature = "gpu")]
pub use evolution::KimuraGpu;
pub use evolution::{detection_power, detection_threshold, error_threshold, kimura_fixation_prob};
#[cfg(feature = "gpu")]
pub use histogram::HistogramGpu;
#[cfg(feature = "gpu")]
pub use hydrology::HargreavesBatchGpu;
#[cfg(feature = "gpu")]
pub use hydrology::{
    Fao56BaseInputs, Fao56Uncertainties, McEt0PropagateGpu, SeasonalGpuParams, SeasonalOutput,
    SeasonalPipelineF64,
};
pub use hydrology::{
    crop_coefficient, fao56_et0, hamon_et0, hargreaves_et0, hargreaves_et0_batch, makkink_et0,
    soil_water_balance, thornthwaite_et0, thornthwaite_heat_index, turc_et0,
};
#[cfg(feature = "gpu")]
pub use jackknife::JackknifeMeanGpu;
pub use jackknife::{JackknifeResult, jackknife, jackknife_mean_variance};
pub use metrics::{
    dot, hill, hill_activation, hill_repression, hit_rate, index_of_agreement, l2_norm, mae, mbe,
    mean, monod, nash_sutcliffe, percentile, r_squared, rmse,
};
pub use moving_window_f64::{MovingWindowResultF64, moving_window_stats_f64};
pub use normal::{norm_cdf, norm_cdf_batch, norm_pdf, norm_pdf_batch, norm_ppf};
pub use regression::{
    FitResult, fit_all, fit_exponential, fit_linear, fit_logarithmic, fit_quadratic,
};
pub use spectral_density::{empirical_spectral_density, marchenko_pastur_bounds};
pub use welford::{WelfordCovState, WelfordState};
