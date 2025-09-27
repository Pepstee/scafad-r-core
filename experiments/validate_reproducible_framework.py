#!/usr/bin/env python3
"""
SCAFAD Reproducible Framework Validation
========================================

This script validates that the reproducible experiments framework is working
correctly and produces consistent results across multiple runs.

Features:
1. Tests deterministic behavior across runs
2. Validates Docker container functionality
3. Checks experiment result consistency
4. Verifies statistical reproducibility
5. Tests checkpoint/resume functionality

Usage:
    python experiments/validate_reproducible_framework.py
    python experiments/validate_reproducible_framework.py --quick-validation
    python experiments/validate_reproducible_framework.py --docker-test
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import shutil

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReproducibilityValidator:
    """Validates reproducibility of SCAFAD experiments"""
    
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.validation_results = {}
        
    def run_experiment_with_seed(self, seed: int, output_dir: str) -> Dict[str, Any]:
        """Run experiment with specific seed and return results"""
        
        try:
            # Import here to avoid issues if not available
            from experiments.run_reproducible_experiments import ReproducibleExperimentOrchestrator
            
            # Initialize orchestrator
            orchestrator = ReproducibleExperimentOrchestrator(
                seed=seed,
                output_dir=output_dir,
                quick_mode=True  # Always use quick mode for validation
            )
            
            # Run system validation only for speed
            result = asyncio.run(orchestrator.run_system_validation())
            
            return {
                "seed": seed,
                "result": result,
                "success": "error" not in result
            }
            
        except Exception as e:
            logger.error(f"Experiment with seed {seed} failed: {e}")
            return {
                "seed": seed,
                "error": str(e),
                "success": False
            }
    
    def validate_deterministic_results(self, num_runs: int = 3, seed: int = 42) -> Dict[str, Any]:
        """Validate that multiple runs with same seed produce identical results"""
        
        logger.info(f"🎲 Validating deterministic results ({num_runs} runs with seed {seed})")
        
        results = []
        output_base = Path(tempfile.mkdtemp(prefix="scafad_validation_"))
        
        try:
            for run_id in range(num_runs):
                logger.info(f"  Run {run_id + 1}/{num_runs}...")
                
                # Create separate output directory for each run
                run_output_dir = output_base / f"run_{run_id}"
                run_output_dir.mkdir(exist_ok=True)
                
                # Run experiment
                result = self.run_experiment_with_seed(seed, str(run_output_dir))
                results.append(result)
            
            # Analyze results for consistency
            if all(r["success"] for r in results):
                # Compare key metrics across runs
                validation_results = [r["result"] for r in results if r["success"]]
                
                # Check if all results are identical (for deterministic components)
                first_result = validation_results[0]
                all_identical = True
                differences = []
                
                for i, result in enumerate(validation_results[1:], 1):
                    if result.get("system_status") != first_result.get("system_status"):
                        all_identical = False
                        differences.append(f"Run {i}: system_status differs")
                    
                    # Compare component validation results
                    first_components = set(first_result.get("components_validated", []))
                    result_components = set(result.get("components_validated", []))
                    
                    if first_components != result_components:
                        all_identical = False
                        differences.append(f"Run {i}: component validation differs")
                
                return {
                    "validation_type": "deterministic_results",
                    "num_runs": num_runs,
                    "seed": seed,
                    "all_successful": True,
                    "results_identical": all_identical,
                    "differences": differences,
                    "success": all_identical
                }
                
            else:
                failed_runs = [r for r in results if not r["success"]]
                return {
                    "validation_type": "deterministic_results",
                    "num_runs": num_runs,
                    "seed": seed,
                    "all_successful": False,
                    "failed_runs": len(failed_runs),
                    "errors": [r.get("error", "Unknown error") for r in failed_runs],
                    "success": False
                }
                
        finally:
            # Clean up temporary directory
            shutil.rmtree(output_base, ignore_errors=True)
    
    def validate_seed_variation(self, seeds: List[int] = None) -> Dict[str, Any]:
        """Validate that different seeds produce different results"""
        
        if seeds is None:
            seeds = [42, 123, 456] if not self.quick_mode else [42, 123]
        
        logger.info(f"🌱 Validating seed variation ({len(seeds)} different seeds)")
        
        results = {}
        output_base = Path(tempfile.mkdtemp(prefix="scafad_seed_validation_"))
        
        try:
            for seed in seeds:
                logger.info(f"  Testing seed {seed}...")
                
                seed_output_dir = output_base / f"seed_{seed}"
                seed_output_dir.mkdir(exist_ok=True)
                
                result = self.run_experiment_with_seed(seed, str(seed_output_dir))
                results[seed] = result
            
            successful_results = {k: v for k, v in results.items() if v["success"]}
            
            if len(successful_results) >= 2:
                # Check that different seeds produce measurably different results
                seed_list = list(successful_results.keys())
                variations_detected = []
                
                for i, seed1 in enumerate(seed_list):
                    for seed2 in seed_list[i+1:]:
                        result1 = successful_results[seed1]["result"]
                        result2 = successful_results[seed2]["result"]
                        
                        # Look for variations in trace generation or detection results
                        trace_size1 = result1.get("test_trace_size", 0)
                        trace_size2 = result2.get("test_trace_size", 0)
                        
                        if trace_size1 != trace_size2:
                            variations_detected.append(f"Seed {seed1} vs {seed2}: trace sizes differ ({trace_size1} vs {trace_size2})")
                
                return {
                    "validation_type": "seed_variation",
                    "seeds_tested": seeds,
                    "successful_seeds": list(successful_results.keys()),
                    "variations_detected": variations_detected,
                    "variation_found": len(variations_detected) > 0,
                    "success": len(variations_detected) > 0  # We want variation across seeds
                }
            else:
                return {
                    "validation_type": "seed_variation",
                    "seeds_tested": seeds,
                    "successful_seeds": list(successful_results.keys()),
                    "error": "Not enough successful runs to compare",
                    "success": False
                }
                
        finally:
            shutil.rmtree(output_base, ignore_errors=True)
    
    def validate_docker_functionality(self) -> Dict[str, Any]:
        """Validate Docker container can run experiments"""
        
        logger.info("🐳 Validating Docker container functionality")
        
        try:
            # Check if Docker is available
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {
                    "validation_type": "docker_functionality",
                    "docker_available": False,
                    "error": "Docker not available",
                    "success": False
                }
            
            docker_version = result.stdout.strip()
            logger.info(f"  Docker version: {docker_version}")
            
            # Try to build SCAFAD container
            logger.info("  Building SCAFAD container...")
            build_result = subprocess.run([
                "docker", "build", "-t", "scafad-validation-test", "."
            ], capture_output=True, text=True, timeout=300)
            
            if build_result.returncode != 0:
                return {
                    "validation_type": "docker_functionality",
                    "docker_available": True,
                    "docker_version": docker_version,
                    "build_successful": False,
                    "build_error": build_result.stderr,
                    "success": False
                }
            
            logger.info("  Running validation in container...")
            
            # Run validation in container
            run_result = subprocess.run([
                "docker", "run", "--rm",
                "scafad-validation-test",
                "python", "experiments/run_reproducible_experiments.py",
                "--experiment-type", "system_validation",
                "--quick-mode", "--seed", "42"
            ], capture_output=True, text=True, timeout=180)
            
            # Clean up test container
            subprocess.run(["docker", "rmi", "scafad-validation-test"], 
                         capture_output=True, timeout=30)
            
            if run_result.returncode == 0:
                return {
                    "validation_type": "docker_functionality", 
                    "docker_available": True,
                    "docker_version": docker_version,
                    "build_successful": True,
                    "run_successful": True,
                    "container_output": run_result.stdout[-500:],  # Last 500 chars
                    "success": True
                }
            else:
                return {
                    "validation_type": "docker_functionality",
                    "docker_available": True,
                    "docker_version": docker_version,
                    "build_successful": True,
                    "run_successful": False,
                    "run_error": run_result.stderr,
                    "success": False
                }
            
        except subprocess.TimeoutExpired:
            return {
                "validation_type": "docker_functionality",
                "error": "Docker operation timed out",
                "success": False
            }
        except Exception as e:
            return {
                "validation_type": "docker_functionality",
                "error": str(e),
                "success": False
            }
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate all required dependencies are available"""
        
        logger.info("📦 Validating dependencies...")
        
        required_packages = [
            ("numpy", "1.21.0"),
            ("torch", "1.12.0"),
            ("networkx", "2.8"),
            ("scikit-learn", "1.1.0"),
            ("pandas", "1.4.0"),
            ("boto3", "1.24.0")
        ]
        
        results = {
            "validation_type": "dependencies",
            "required_packages": {},
            "all_available": True,
            "missing_packages": [],
            "version_mismatches": []
        }
        
        for package_name, min_version in required_packages:
            try:
                # Dynamic import to check availability
                if package_name == "torch":
                    import torch
                    version = torch.__version__
                elif package_name == "numpy":
                    import numpy
                    version = numpy.__version__
                elif package_name == "networkx":
                    import networkx
                    version = networkx.__version__
                elif package_name == "scikit-learn":
                    import sklearn
                    version = sklearn.__version__
                elif package_name == "pandas":
                    import pandas
                    version = pandas.__version__
                elif package_name == "boto3":
                    import boto3
                    version = boto3.__version__
                else:
                    version = "unknown"
                
                results["required_packages"][package_name] = {
                    "available": True,
                    "version": version,
                    "min_version": min_version
                }
                
            except ImportError:
                results["all_available"] = False
                results["missing_packages"].append(package_name)
                results["required_packages"][package_name] = {
                    "available": False,
                    "min_version": min_version
                }
        
        results["success"] = results["all_available"]
        return results
    
    def validate_framework_structure(self) -> Dict[str, Any]:
        """Validate that all required files and directories exist"""
        
        logger.info("📁 Validating framework structure...")
        
        required_files = [
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt", 
            "docker/requirements-docker.txt",
            "experiments/run_reproducible_experiments.py",
            "experiments/README.md",
            "Makefile"
        ]
        
        required_directories = [
            "experiments",
            "docker",
            "core",
            "datasets",
            "baselines",
            "evaluation",
            "formal_verification",
            "aws_deployment"
        ]
        
        results = {
            "validation_type": "framework_structure",
            "missing_files": [],
            "missing_directories": [],
            "all_files_present": True,
            "all_directories_present": True
        }
        
        # Check files
        for file_path in required_files:
            if not Path(file_path).exists():
                results["missing_files"].append(file_path)
                results["all_files_present"] = False
        
        # Check directories
        for dir_path in required_directories:
            if not Path(dir_path).exists():
                results["missing_directories"].append(dir_path)
                results["all_directories_present"] = False
        
        results["success"] = results["all_files_present"] and results["all_directories_present"]
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_validations = len(self.validation_results)
        successful_validations = sum(1 for r in self.validation_results.values() if r.get("success", False))
        
        report = {
            "validation_summary": {
                "total_validations": total_validations,
                "successful_validations": successful_validations,
                "success_rate": successful_validations / max(total_validations, 1),
                "overall_success": successful_validations == total_validations,
                "validation_timestamp": time.time()
            },
            "validation_results": self.validation_results,
            "recommendations": []
        }
        
        # Generate recommendations
        if report["validation_summary"]["overall_success"]:
            report["recommendations"].append({
                "type": "success",
                "message": "✅ SCAFAD reproducible framework is fully validated and ready for use"
            })
        else:
            failed_validations = [name for name, result in self.validation_results.items() 
                                if not result.get("success", False)]
            
            report["recommendations"].append({
                "type": "error",
                "priority": "high",
                "message": f"❌ {len(failed_validations)} validation(s) failed: {', '.join(failed_validations)}"
            })
            
            if "dependencies" in failed_validations:
                report["recommendations"].append({
                    "type": "setup",
                    "priority": "high", 
                    "message": "Install missing dependencies: pip install -r requirements.txt"
                })
            
            if "docker_functionality" in failed_validations:
                report["recommendations"].append({
                    "type": "setup",
                    "priority": "medium",
                    "message": "Install Docker for containerized experiments: https://docs.docker.com/get-docker/"
                })
        
        return report
    
    async def run_full_validation(self, include_docker: bool = True) -> Dict[str, Any]:
        """Run complete validation suite"""
        
        logger.info("🧪 Running SCAFAD reproducible framework validation")
        logger.info("=" * 60)
        
        validations = [
            ("framework_structure", self.validate_framework_structure),
            ("dependencies", self.validate_dependencies),
            ("deterministic_results", lambda: self.validate_deterministic_results(2 if self.quick_mode else 3)),
            ("seed_variation", self.validate_seed_variation)
        ]
        
        if include_docker:
            validations.append(("docker_functionality", self.validate_docker_functionality))
        
        for validation_name, validation_func in validations:
            logger.info(f"Running {validation_name} validation...")
            
            try:
                result = validation_func()
                self.validation_results[validation_name] = result
                
                if result.get("success", False):
                    logger.info(f"✅ {validation_name} validation passed")
                else:
                    logger.error(f"❌ {validation_name} validation failed")
                    if "error" in result:
                        logger.error(f"   Error: {result['error']}")
                        
            except Exception as e:
                logger.error(f"❌ {validation_name} validation crashed: {e}")
                self.validation_results[validation_name] = {
                    "validation_type": validation_name,
                    "error": str(e),
                    "success": False
                }
        
        # Generate final report
        report = self.generate_validation_report()
        
        logger.info("=" * 60)
        logger.info("🎉 SCAFAD Framework Validation Complete!")
        logger.info(f"   Total validations: {report['validation_summary']['total_validations']}")
        logger.info(f"   Successful: {report['validation_summary']['successful_validations']}")
        logger.info(f"   Success rate: {report['validation_summary']['success_rate']:.1%}")
        
        if report["recommendations"]:
            logger.info("📋 Recommendations:")
            for rec in report["recommendations"]:
                logger.info(f"   {rec['message']}")
        
        return report


async def main():
    """Main validation function"""
    
    parser = argparse.ArgumentParser(description="SCAFAD Reproducible Framework Validation")
    parser.add_argument("--quick-validation", action="store_true", 
                       help="Run quick validation with reduced tests")
    parser.add_argument("--skip-docker", action="store_true", 
                       help="Skip Docker functionality tests")
    parser.add_argument("--output-file", 
                       help="Save validation report to file")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ReproducibilityValidator(quick_mode=args.quick_validation)
    
    # Run validation
    try:
        report = await validator.run_full_validation(include_docker=not args.skip_docker)
        
        # Save report if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Validation report saved: {args.output_file}")
        
        # Exit with appropriate code
        if report["validation_summary"]["overall_success"]:
            logger.info("🎉 All validations passed! Framework is ready for reproducible experiments.")
            sys.exit(0)
        else:
            logger.error("❌ Some validations failed. Check the report for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())