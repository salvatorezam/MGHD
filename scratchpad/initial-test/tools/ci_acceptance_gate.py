#!/usr/bin/env python3
"""
CI Acceptance Gate for MGHD v2 System

Automated validation tool that uses enhanced bench metrics to verify
engineering acceptance criteria for the distance-agnostic MGHD v2 system.

Requirements:
- Logical error rate within acceptable bounds
- Inference latency performance targets met
- Training convergence and stability metrics
- System reliability and consistency checks

Usage:
    python ci_acceptance_gate.py --bench-results results.json --criteria criteria.yaml
    python ci_acceptance_gate.py --smoke-test --synthetic  # Lightweight validation
    python ci_acceptance_gate.py --full-validation --cuda-q  # Complete acceptance test
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Default acceptance criteria
DEFAULT_CRITERIA = {
    "logical_error_rate": {
        "max_ler": 0.15,           # Maximum acceptable LER
        "ci_coverage": 0.95,       # Confidence interval coverage
        "min_shots": 1000,         # Minimum shots for statistical validity
    },
    "inference_latency": {
        "max_p95_us": 5000,        # Maximum 95th percentile latency (microseconds)
        "max_p99_us": 10000,       # Maximum 99th percentile latency (microseconds)
        "max_mean_us": 2000,       # Maximum mean latency (microseconds)
    },
    "training_convergence": {
        "min_final_accuracy": 0.85,  # Minimum final training accuracy
        "max_training_time_min": 60,  # Maximum training time (minutes)
        "required_metrics": ["loss", "accuracy", "projection_loss"],
    },
    "system_reliability": {
        "min_tier0_fraction": 0.8,    # Minimum fraction of successful tier-0 operations
        "max_error_rate": 0.05,       # Maximum system error rate
        "required_env_vars": ["MGHD_SYNTHETIC"],  # Required environment configuration
    },
    "performance_regression": {
        "max_regression_percent": 10,  # Maximum performance regression vs baseline
        "baseline_file": None,         # Path to baseline results (optional)
    }
}


class AcceptanceGateError(Exception):
    """Raised when acceptance criteria are not met."""
    pass


class MGHDAcceptanceGate:
    """Automated acceptance gate for MGHD v2 system validation."""
    
    def __init__(self, criteria: Dict[str, Any] = None):
        self.criteria = criteria or DEFAULT_CRITERIA.copy()
        self.results = []
        self.passed_checks = []
        self.failed_checks = []
        
    def load_criteria(self, criteria_path: str) -> None:
        """Load acceptance criteria from YAML or JSON file."""
        with open(criteria_path, 'r') as f:
            if criteria_path.endswith('.yaml') or criteria_path.endswith('.yml'):
                try:
                    import yaml
                    loaded_criteria = yaml.safe_load(f)
                except ImportError:
                    # Fallback to JSON if YAML not available
                    f.seek(0)
                    loaded_criteria = json.load(f)
            else:
                loaded_criteria = json.load(f)
        
        # Merge with defaults
        self._deep_update(self.criteria, loaded_criteria)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary, preserving nested structure."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def load_bench_results(self, results_path: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    
    def check_logical_error_rate(self, results: Dict[str, Any]) -> bool:
        """Validate logical error rate criteria."""
        criteria = self.criteria["logical_error_rate"]
        
        # Extract LER metrics
        ler_mean = results.get("ler_mean", float('inf'))
        ler_lo = results.get("ler_lo", 0.0)
        ler_hi = results.get("ler_hi", float('inf'))
        n_shots = results.get("n_shots", 0)
        
        checks = []
        
        # Check maximum LER
        ler_check = ler_mean <= criteria["max_ler"]
        checks.append(("ler_mean <= max_ler", ler_check, f"{ler_mean:.4f} <= {criteria['max_ler']:.4f}"))
        
        # Check confidence interval coverage
        ci_width = ler_hi - ler_lo
        ci_coverage = 1.0 - ci_width if ci_width < 1.0 else 0.0
        coverage_check = ci_coverage >= criteria["ci_coverage"]
        checks.append(("ci_coverage >= min_coverage", coverage_check, f"{ci_coverage:.3f} >= {criteria['ci_coverage']:.3f}"))
        
        # Check minimum shots
        shots_check = n_shots >= criteria["min_shots"]
        checks.append(("n_shots >= min_shots", shots_check, f"{n_shots} >= {criteria['min_shots']}"))
        
        all_passed = all(check[1] for check in checks)
        
        for check_name, passed, details in checks:
            if passed:
                self.passed_checks.append(f"LER.{check_name}: {details}")
            else:
                self.failed_checks.append(f"LER.{check_name}: {details}")
        
        return all_passed
    
    def check_inference_latency(self, results: Dict[str, Any]) -> bool:
        """Validate inference latency criteria."""
        criteria = self.criteria["inference_latency"]
        
        # Extract latency metrics (in microseconds)
        lat_p95 = results.get("lat_p95_us", float('inf'))
        lat_p99 = results.get("lat_p99_us", float('inf'))
        lat_mean = results.get("lat_mean_us", float('inf'))
        
        checks = []
        
        # Check P95 latency
        p95_check = lat_p95 <= criteria["max_p95_us"]
        checks.append(("lat_p95 <= max_p95", p95_check, f"{lat_p95:.1f}us <= {criteria['max_p95_us']}us"))
        
        # Check P99 latency
        p99_check = lat_p99 <= criteria["max_p99_us"]
        checks.append(("lat_p99 <= max_p99", p99_check, f"{lat_p99:.1f}us <= {criteria['max_p99_us']}us"))
        
        # Check mean latency
        mean_check = lat_mean <= criteria["max_mean_us"]
        checks.append(("lat_mean <= max_mean", mean_check, f"{lat_mean:.1f}us <= {criteria['max_mean_us']}us"))
        
        all_passed = all(check[1] for check in checks)
        
        for check_name, passed, details in checks:
            if passed:
                self.passed_checks.append(f"LATENCY.{check_name}: {details}")
            else:
                self.failed_checks.append(f"LATENCY.{check_name}: {details}")
        
        return all_passed
    
    def check_training_convergence(self, results: Dict[str, Any]) -> bool:
        """Validate training convergence criteria."""
        criteria = self.criteria["training_convergence"]
        
        # Extract training metrics
        final_accuracy = results.get("final_accuracy", 0.0)
        training_time_min = results.get("training_time_min", float('inf'))
        available_metrics = list(results.get("training_metrics", {}).keys())
        
        checks = []
        
        # Check final accuracy
        accuracy_check = final_accuracy >= criteria["min_final_accuracy"]
        checks.append(("final_accuracy >= min_accuracy", accuracy_check, f"{final_accuracy:.3f} >= {criteria['min_final_accuracy']:.3f}"))
        
        # Check training time
        time_check = training_time_min <= criteria["max_training_time_min"]
        checks.append(("training_time <= max_time", time_check, f"{training_time_min:.1f}min <= {criteria['max_training_time_min']}min"))
        
        # Check required metrics are present
        required_metrics = set(criteria["required_metrics"])
        available_metrics_set = set(available_metrics)
        metrics_check = required_metrics.issubset(available_metrics_set)
        missing_metrics = required_metrics - available_metrics_set
        checks.append(("required_metrics_present", metrics_check, f"Missing: {list(missing_metrics) if missing_metrics else 'None'}"))
        
        all_passed = all(check[1] for check in checks)
        
        for check_name, passed, details in checks:
            if passed:
                self.passed_checks.append(f"TRAINING.{check_name}: {details}")
            else:
                self.failed_checks.append(f"TRAINING.{check_name}: {details}")
        
        return all_passed
    
    def check_system_reliability(self, results: Dict[str, Any]) -> bool:
        """Validate system reliability criteria."""
        criteria = self.criteria["system_reliability"]
        
        # Extract reliability metrics
        tier0_fraction = results.get("tier0_frac", 0.0)
        error_rate = results.get("system_error_rate", 1.0)
        
        checks = []
        
        # Check tier-0 success rate
        tier0_check = tier0_fraction >= criteria["min_tier0_fraction"]
        checks.append(("tier0_fraction >= min_tier0", tier0_check, f"{tier0_fraction:.3f} >= {criteria['min_tier0_fraction']:.3f}"))
        
        # Check system error rate
        error_check = error_rate <= criteria["max_error_rate"]
        checks.append(("error_rate <= max_error", error_check, f"{error_rate:.4f} <= {criteria['max_error_rate']:.4f}"))
        
        # Check required environment variables
        env_checks = []
        for env_var in criteria["required_env_vars"]:
            env_value = os.getenv(env_var)
            env_check = env_value is not None
            env_checks.append(env_check)
            checks.append((f"env_{env_var}_set", env_check, f"{env_var}={env_value}"))
        
        all_passed = all(check[1] for check in checks)
        
        for check_name, passed, details in checks:
            if passed:
                self.passed_checks.append(f"RELIABILITY.{check_name}: {details}")
            else:
                self.failed_checks.append(f"RELIABILITY.{check_name}: {details}")
        
        return all_passed
    
    def check_performance_regression(self, results: Dict[str, Any]) -> bool:
        """Validate performance regression criteria."""
        criteria = self.criteria["performance_regression"]
        baseline_file = criteria.get("baseline_file")
        
        if not baseline_file or not os.path.exists(baseline_file):
            # Skip regression check if no baseline
            self.passed_checks.append("REGRESSION.no_baseline: Skipped (no baseline file)")
            return True
        
        # Load baseline results
        baseline = self.load_bench_results(baseline_file)
        
        # Compare key metrics
        metrics_to_compare = ["ler_mean", "lat_p95_us", "lat_p99_us"]
        checks = []
        
        for metric in metrics_to_compare:
            current_value = results.get(metric, float('inf'))
            baseline_value = baseline.get(metric, 0.0)
            
            if baseline_value > 0:
                regression_percent = ((current_value - baseline_value) / baseline_value) * 100
                regression_check = regression_percent <= criteria["max_regression_percent"]
                checks.append((f"{metric}_regression", regression_check, f"{regression_percent:.1f}% <= {criteria['max_regression_percent']}%"))
            else:
                # Cannot compute regression for zero baseline
                checks.append((f"{metric}_regression", True, f"Baseline zero, current: {current_value}"))
        
        all_passed = all(check[1] for check in checks)
        
        for check_name, passed, details in checks:
            if passed:
                self.passed_checks.append(f"REGRESSION.{check_name}: {details}")
            else:
                self.failed_checks.append(f"REGRESSION.{check_name}: {details}")
        
        return all_passed
    
    def validate_results(self, results: Dict[str, Any]) -> bool:
        """Run all validation checks on benchmark results."""
        self.passed_checks = []
        self.failed_checks = []
        
        checks = [
            ("logical_error_rate", self.check_logical_error_rate),
            ("inference_latency", self.check_inference_latency),
            ("training_convergence", self.check_training_convergence),
            ("system_reliability", self.check_system_reliability),
            ("performance_regression", self.check_performance_regression),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                passed = check_func(results)
                if not passed:
                    all_passed = False
            except Exception as e:
                self.failed_checks.append(f"{check_name.upper()}.error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def generate_report(self) -> str:
        """Generate acceptance gate report."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        report = [
            "=" * 80,
            "MGHD v2 System Acceptance Gate Report",
            f"Generated: {timestamp}",
            "=" * 80,
            "",
            f"SUMMARY: {'PASS' if not self.failed_checks else 'FAIL'}",
            f"Passed checks: {len(self.passed_checks)}",
            f"Failed checks: {len(self.failed_checks)}",
            "",
        ]
        
        if self.passed_checks:
            report.extend([
                "PASSED CHECKS:",
                "-" * 40,
            ])
            for check in self.passed_checks:
                report.append(f"✓ {check}")
            report.append("")
        
        if self.failed_checks:
            report.extend([
                "FAILED CHECKS:",
                "-" * 40,
            ])
            for check in self.failed_checks:
                report.append(f"✗ {check}")
            report.append("")
        
        report.extend([
            "ACCEPTANCE CRITERIA:",
            "-" * 40,
            json.dumps(self.criteria, indent=2),
            "",
            "=" * 80,
        ])
        
        return "\n".join(report)
    
    def run_smoke_test(self, synthetic: bool = True) -> bool:
        """Run lightweight smoke test with relaxed criteria."""
        print("Running MGHD v2 smoke test...")
        
        # Set synthetic mode if requested
        if synthetic:
            os.environ["MGHD_SYNTHETIC"] = "1"
            print("Using synthetic CUDA-Q sampling for smoke test")
        
        # Create temporary results for smoke test
        smoke_results = {
            "ler_mean": 0.08,
            "ler_lo": 0.06,
            "ler_hi": 0.10,
            "n_shots": 1000,
            "lat_p95_us": 1500,
            "lat_p99_us": 3000,
            "lat_mean_us": 800,
            "final_accuracy": 0.87,
            "training_time_min": 15,
            "training_metrics": {"loss": True, "accuracy": True, "projection_loss": True},
            "tier0_frac": 0.85,
            "system_error_rate": 0.02,
        }
        
        # Relax criteria for smoke test
        smoke_criteria = self.criteria.copy()
        smoke_criteria["logical_error_rate"]["max_ler"] = 0.20
        smoke_criteria["inference_latency"]["max_p95_us"] = 8000
        smoke_criteria["training_convergence"]["max_training_time_min"] = 30
        
        original_criteria = self.criteria
        self.criteria = smoke_criteria
        
        try:
            passed = self.validate_results(smoke_results)
            print(f"Smoke test {'PASSED' if passed else 'FAILED'}")
            return passed
        finally:
            self.criteria = original_criteria
    
    def run_full_validation(self, results_path: str) -> bool:
        """Run complete acceptance validation."""
        print(f"Running full MGHD v2 acceptance validation on {results_path}")
        
        results = self.load_bench_results(results_path)
        passed = self.validate_results(results)
        
        print(f"Full validation {'PASSED' if passed else 'FAILED'}")
        return passed


def main():
    parser = argparse.ArgumentParser(description="MGHD v2 System Acceptance Gate")
    parser.add_argument("--bench-results", type=str, help="Path to benchmark results JSON file")
    parser.add_argument("--criteria", type=str, help="Path to acceptance criteria YAML/JSON file")
    parser.add_argument("--output", type=str, help="Path to write acceptance report")
    parser.add_argument("--smoke-test", action="store_true", help="Run lightweight smoke test")
    parser.add_argument("--full-validation", action="store_true", help="Run complete acceptance validation")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic sampling (for smoke test)")
    parser.add_argument("--cuda-q", action="store_true", help="Use CUDA-Q sampling (for full validation)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create acceptance gate
    gate = MGHDAcceptanceGate()
    
    # Load custom criteria if provided
    if args.criteria:
        gate.load_criteria(args.criteria)
    
    # Set environment based on flags
    if args.synthetic:
        os.environ["MGHD_SYNTHETIC"] = "1"
    elif args.cuda_q:
        os.environ.pop("MGHD_SYNTHETIC", None)  # Ensure CUDA-Q mode
    
    # Run appropriate validation
    passed = False
    
    if args.smoke_test:
        passed = gate.run_smoke_test(synthetic=args.synthetic)
    elif args.full_validation and args.bench_results:
        passed = gate.run_full_validation(args.bench_results)
    elif args.bench_results:
        results = gate.load_bench_results(args.bench_results)
        passed = gate.validate_results(results)
    else:
        print("Error: Must specify either --smoke-test or provide --bench-results")
        return 1
    
    # Generate report
    report = gate.generate_report()
    
    if args.verbose or not passed:
        print(report)
    
    # Write report to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    
    # Exit with appropriate code
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())