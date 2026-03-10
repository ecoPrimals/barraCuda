// SPDX-License-Identifier: AGPL-3.0-only
//! Markdown report generation for cross-spring shader evolution.

use std::fmt::Write;

use super::registry::{EVOLUTION_TIMELINE, REGISTRY, cross_spring_matrix, shaders_from};
use super::types::SpringDomain as SD;

/// Generate a human-readable cross-spring evolution report.
#[must_use]
pub fn evolution_report() -> String {
    let mut report = String::from("# Cross-Spring Shader Evolution Report\n\n");

    report.push_str("## Timeline\n\n");
    for event in EVOLUTION_TIMELINE.iter() {
        let _ = write!(
            report,
            "**{}** — {} →\n  {}\n\n",
            event.date, event.from, event.description
        );
    }

    report.push_str("## Dependency Matrix (shader count)\n\n");
    report.push_str(
        "| From \\ To | hotSpring | wetSpring | neuralSpring | airSpring | groundSpring |\n",
    );
    report.push_str(
        "|-----------|-----------|-----------|--------------|-----------|---------------|\n",
    );

    let matrix = cross_spring_matrix();
    let domains = [
        SD::HOT_SPRING,
        SD::WET_SPRING,
        SD::NEURAL_SPRING,
        SD::AIR_SPRING,
        SD::GROUND_SPRING,
    ];
    for from in &domains {
        let _ = write!(report, "| **{from}** ");
        for to in &domains {
            let count = matrix.get(&(*from, *to)).copied().unwrap_or(0);
            if from == to {
                report.push_str("| — ");
            } else if count > 0 {
                let _ = write!(report, "| {count} ");
            } else {
                report.push_str("| · ");
            }
        }
        report.push_str("|\n");
    }

    report.push_str("\n## Shader Categories by Origin\n\n");
    for domain in &domains {
        let shaders = shaders_from(*domain);
        if shaders.is_empty() {
            continue;
        }
        let _ = write!(report, "### {} ({} shaders)\n\n", domain, shaders.len());
        for s in &shaders {
            let cross = s.consumers.iter().filter(|c| **c != s.origin).count();
            let _ = write!(
                report,
                "- `{}` [{}] — {} cross-spring consumer{}\n  Created: {} | Absorbed: {}\n",
                s.path,
                s.category,
                cross,
                if cross == 1 { "" } else { "s" },
                s.created,
                s.absorbed,
            );
        }
        report.push('\n');
    }

    report
}

/// Total number of shaders in the registry.
#[must_use]
pub fn shader_count() -> usize {
    REGISTRY.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evolution_report_contains_key_sections() {
        let report = evolution_report();
        assert!(report.contains("Timeline"));
        assert!(report.contains("Dependency Matrix"));
        assert!(report.contains("hotSpring"));
        assert!(report.contains("neuralSpring"));
        assert!(report.contains("wetSpring"));
    }

    #[test]
    fn shader_count_matches_registry() {
        assert!(shader_count() >= 27);
    }
}
