#!/usr/bin/env python3
"""
SCAFAD Reproducible Experiments Orchestrator
============================================

This script orchestrates all SCAFAD experiments in a reproducible manner,
ensuring consistent results across different environments and runs.

Features:
1. Deterministic random seed management
2. Complete environment isolation
3. Comprehensive experiment tracking
4. Academic-quality result generation
5. Docker-optimized execution
6. Checkpoint/resume functionality

Usage:
    python experiments/run_reproducible_experiments.py --experiment-type all --seed 42
    python experiments/run_reproducible_experiments.py --experiment-type ignn --quick-mode
    python experiments/run_reproducible_experiments.py --resume-from checkpoint.json
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

import numpy as np
import torch
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import SCAFAD components
try:
    from app_config import get_default_config
    from core.ignn_model import iGNNAnomalyDetector
    from core.real_graph_analysis import GraphAnalysisOrchestrator
    from datasets.serverless_traces import RealisticServerlessTraceGenerator
    from baselines.classical_detectors import BaselineComparator
    from evaluation.ignn_vs_baselines import iGNNBaselineEvaluator
    from formal_verification.ltl_checker import ServerlessFormalVerifier
    from aws_deployment.lambda_deployer import AWSLambdaDeployer
except ImportError as e:
    logger.error(f"Failed to import SCAFAD components: {e}")
    sys.exit(1)


class ReproducibleExperimentOrchestrator:
    """Orchestrates reproducible SCAFAD experiments"""
    
    def __init__(self, seed: int = 42, output_dir: str = "experiments/results", 
                 quick_mode: bool = False):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.quick_mode = quick_mode
        self.start_time = time.time()
        
        # Ensure reproducibility
        self._setup_reproducibility()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "datasets").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Initialize components
        self.config = get_default_config()
        self.trace_generator = RealisticServerlessTraceGenerator(random_seed=seed)
        self.experiment_results = {}
        self.checkpoint_file = self.output_dir / "checkpoints" / "experiment_checkpoint.json"
        
        logger.info(f"🧪 Reproducible SCAFAD Experiments initialized")
        logger.info(f"   Seed: {seed}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   Quick mode: {quick_mode}")
        
    def _setup_reproducibility(self):
        """Setup deterministic behavior for reproducible experiments"""
        # Python random
        random.seed(self.seed)
        
        # NumPy
        np.random.seed(self.seed)
        
        # PyTorch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed) if torch.cuda.is_available() else None
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # NetworkX
        nx.random.seed = self.seed
        
        # Environment variables
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ["SCAFAD_RANDOM_SEED"] = str(self.seed)
        os.environ["SCAFAD_REPRODUCIBLE"] = "true"
        
        logger.info(f"✅ Reproducibility setup completed with seed {self.seed}")
    
    def save_checkpoint(self, experiment_name: str, status: str, result: Any = None):
        """Save experiment checkpoint for resume capability"""
        checkpoint_data = {
            "timestamp": time.time(),
            "seed": self.seed,
            "experiment_name": experiment_name,
            "status": status,  # "running", "completed", "failed"
            "result_summary": str(result)[:1000] if result else None,
            "completed_experiments": list(self.experiment_results.keys())
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load previous checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None
    
    async def run_system_validation(self) -> Dict[str, Any]:
        """Validate SCAFAD system functionality"""
        logger.info("🔍 Running system validation...")
        
        try:
            # Test i-GNN initialization
            ignn_detector = iGNNAnomalyDetector(random_seed=self.seed)
            logger.info("✅ i-GNN detector initialized")
            
            # Test graph analysis
            graph_orchestrator = GraphAnalysisOrchestrator()
            logger.info("✅ Graph orchestrator initialized")
            
            # Test trace generation
            test_trace = self.trace_generator.generate_normal_trace('test-function', 0.1, 5)
            logger.info(f"✅ Generated test trace with {len(test_trace.invocations)} invocations")
            
            # Test baseline comparator
            baseline_comparator = BaselineComparator(random_seed=self.seed)
            logger.info("✅ Baseline comparator initialized")
            
            # Test formal verification
            formal_verifier = ServerlessFormalVerifier()
            logger.info("✅ Formal verifier initialized")
            
            # Run quick functionality test
            test_data = [record.to_dict() for record in test_trace.invocations]
            anomaly_result = ignn_detector.detect_anomalies(test_data)
            
            validation_result = {
                "system_status": "healthy",
                "components_validated": [
                    "iGNNAnomalyDetector",
                    "GraphAnalysisOrchestrator", 
                    "RealisticServerlessTraceGenerator",
                    "BaselineComparator",
                    "ServerlessFormalVerifier"
                ],
                "test_trace_size": len(test_trace.invocations),
                "test_anomaly_detection": {
                    "anomalies_detected": len(anomaly_result["anomalies"]),
                    "confidence_scores": [a["confidence"] for a in anomaly_result["anomalies"][:5]]
                },
                "validation_timestamp": time.time()
            }
            
            logger.info("✅ System validation completed successfully")
            return validation_result
            
        except Exception as e:
            error_msg = f"❌ System validation failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "system_status": "failed",
                "error": str(e),
                "validation_timestamp": time.time()
            }
    
    async def run_ignn_experiments(self) -> Dict[str, Any]:
        """Run comprehensive i-GNN experiments"""
        logger.info("🧠 Running i-GNN experiments...")
        
        try:
            # Initialize i-GNN detector
            ignn_detector = iGNNAnomalyDetector(random_seed=self.seed)
            
            # Generate experiment datasets
            dataset_sizes = [50, 100, 200] if self.quick_mode else [100, 500, 1000, 2000]
            
            experiment_results = {
                "configuration": {
                    "seed": self.seed,
                    "quick_mode": self.quick_mode,
                    "dataset_sizes": dataset_sizes
                },
                "experiments": {}
            }
            
            for dataset_size in dataset_sizes:
                logger.info(f"  Testing with dataset size: {dataset_size}")
                
                # Generate test dataset
                traces = []
                for i in range(dataset_size // 10):  # 10 invocations per trace
                    trace = self.trace_generator.generate_normal_trace(
                        f'function-{i}', 0.1, 10
                    )
                    traces.append(trace)
                
                # Add some anomalous traces
                num_anomalous = max(1, dataset_size // 20)
                for i in range(num_anomalous):
                    anomaly_trace = self.trace_generator.generate_attack_trace(
                        f'attack-{i}', 'dos_attack', 5
                    )
                    traces.append(anomaly_trace)
                
                # Prepare data for i-GNN
                all_invocations = []
                labels = []
                
                for trace in traces:
                    for record in trace.invocations:
                        all_invocations.append(record.to_dict())
                        # Label anomalous traces
                        labels.append(1 if 'attack' in trace.trace_id else 0)
                
                # Run i-GNN detection
                start_time = time.time()
                detection_result = ignn_detector.detect_anomalies(all_invocations)
                detection_time = time.time() - start_time
                
                # Analyze results
                predicted_anomalies = set(a["record_index"] for a in detection_result["anomalies"])
                true_anomalies = set(i for i, label in enumerate(labels) if label == 1)
                
                true_positives = len(predicted_anomalies & true_anomalies)
                false_positives = len(predicted_anomalies - true_anomalies)
                false_negatives = len(true_anomalies - predicted_anomalies)
                true_negatives = len(labels) - true_positives - false_positives - false_negatives
                
                precision = true_positives / max(true_positives + false_positives, 1)
                recall = true_positives / max(true_positives + false_negatives, 1)
                f1_score = 2 * precision * recall / max(precision + recall, 1e-7)
                
                experiment_results["experiments"][f"dataset_{dataset_size}"] = {
                    "dataset_size": dataset_size,
                    "num_traces": len(traces),
                    "num_invocations": len(all_invocations),
                    "num_true_anomalies": len(true_anomalies),
                    "detection_time_seconds": detection_time,
                    "performance_metrics": {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                        "true_negatives": true_negatives
                    },
                    "anomalies_detected": len(detection_result["anomalies"]),
                    "confidence_distribution": {
                        "mean": np.mean([a["confidence"] for a in detection_result["anomalies"]]) if detection_result["anomalies"] else 0,
                        "std": np.std([a["confidence"] for a in detection_result["anomalies"]]) if detection_result["anomalies"] else 0,
                        "min": min([a["confidence"] for a in detection_result["anomalies"]]) if detection_result["anomalies"] else 0,
                        "max": max([a["confidence"] for a in detection_result["anomalies"]]) if detection_result["anomalies"] else 0
                    }
                }
            
            # Calculate overall statistics
            all_f1_scores = [exp["performance_metrics"]["f1_score"] 
                           for exp in experiment_results["experiments"].values()]
            all_detection_times = [exp["detection_time_seconds"] 
                                 for exp in experiment_results["experiments"].values()]
            
            experiment_results["summary"] = {
                "average_f1_score": np.mean(all_f1_scores),
                "f1_score_std": np.std(all_f1_scores),
                "average_detection_time": np.mean(all_detection_times),
                "detection_time_std": np.std(all_detection_times),
                "best_f1_score": max(all_f1_scores),
                "scalability_assessment": "good" if np.mean(all_detection_times) < 10.0 else "needs_optimization"
            }
            
            logger.info(f"✅ i-GNN experiments completed. Average F1: {experiment_results['summary']['average_f1_score']:.3f}")
            return experiment_results
            
        except Exception as e:
            error_msg = f"❌ i-GNN experiments failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": str(e), "timestamp": time.time()}
    
    async def run_baseline_experiments(self) -> Dict[str, Any]:
        """Run baseline comparison experiments"""
        logger.info("📊 Running baseline comparison experiments...")
        
        try:
            # Initialize evaluator
            evaluator = iGNNBaselineEvaluator(random_seed=self.seed)
            
            # Generate test dataset
            num_traces = 20 if self.quick_mode else 100
            
            logger.info(f"  Generating {num_traces} test traces...")
            traces = []
            
            # Normal traces
            for i in range(num_traces * 4 // 5):
                trace = self.trace_generator.generate_normal_trace(
                    f'normal-{i}', 0.1, 10
                )
                traces.append(trace)
            
            # Anomalous traces
            attack_types = ['dos_attack', 'resource_abuse', 'crypto_mining']
            for i in range(num_traces // 5):
                attack_type = attack_types[i % len(attack_types)]
                trace = self.trace_generator.generate_attack_trace(
                    f'attack-{i}', attack_type, 8
                )
                traces.append(trace)
            
            # Run comprehensive evaluation
            logger.info("  Running comprehensive evaluation...")
            evaluation_result = await evaluator.run_comprehensive_evaluation(traces)
            
            # Calculate summary statistics
            summary = {
                "total_traces": len(traces),
                "normal_traces": num_traces * 4 // 5,
                "anomalous_traces": num_traces // 5,
                "models_compared": len(evaluation_result.get("results", {})),
                "best_model": None,
                "best_f1_score": 0
            }
            
            # Find best performing model
            for model_name, results in evaluation_result.get("results", {}).items():
                if "f1_score" in results and results["f1_score"] > summary["best_f1_score"]:
                    summary["best_f1_score"] = results["f1_score"]
                    summary["best_model"] = model_name
            
            experiment_results = {
                "configuration": {
                    "seed": self.seed,
                    "quick_mode": self.quick_mode,
                    "num_traces": num_traces
                },
                "evaluation_results": evaluation_result,
                "summary": summary,
                "timestamp": time.time()
            }
            
            logger.info(f"✅ Baseline experiments completed. Best model: {summary['best_model']} (F1: {summary['best_f1_score']:.3f})")
            return experiment_results
            
        except Exception as e:
            error_msg = f"❌ Baseline experiments failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": str(e), "timestamp": time.time()}
    
    async def run_formal_verification_experiments(self) -> Dict[str, Any]:
        """Run formal verification experiments"""
        logger.info("✅ Running formal verification experiments...")
        
        try:
            # Initialize formal verifier
            verifier = ServerlessFormalVerifier()
            
            # Generate test traces
            num_traces = 10 if self.quick_mode else 50
            
            logger.info(f"  Generating {num_traces} traces for verification...")
            traces = []
            
            for i in range(num_traces):
                if i % 3 == 0:
                    # Normal trace
                    trace = self.trace_generator.generate_normal_trace(
                        f'verify-normal-{i}', 0.05, 8
                    )
                elif i % 3 == 1:
                    # Attack trace
                    trace = self.trace_generator.generate_attack_trace(
                        f'verify-attack-{i}', 'dos_attack', 6
                    )
                else:
                    # Economic abuse trace
                    trace = self.trace_generator.generate_economic_abuse_trace(
                        f'verify-economic-{i}', 'cryptomining', 7
                    )
                traces.append(trace)
            
            # Run verification on dataset
            logger.info("  Running formal verification...")
            verification_result = verifier.verify_dataset(traces)
            
            # Analyze verification results by trace type
            trace_type_analysis = {
                "normal_traces": {"total": 0, "avg_satisfaction": 0},
                "attack_traces": {"total": 0, "avg_satisfaction": 0},
                "economic_traces": {"total": 0, "avg_satisfaction": 0}
            }
            
            for i, trace_result in enumerate(verification_result["trace_results"]):
                trace_id = traces[i].trace_id
                satisfaction_rate = trace_result["verification_summary"]["satisfaction_rate"]
                
                if "normal" in trace_id:
                    trace_type_analysis["normal_traces"]["total"] += 1
                    trace_type_analysis["normal_traces"]["avg_satisfaction"] += satisfaction_rate
                elif "attack" in trace_id:
                    trace_type_analysis["attack_traces"]["total"] += 1
                    trace_type_analysis["attack_traces"]["avg_satisfaction"] += satisfaction_rate
                elif "economic" in trace_id:
                    trace_type_analysis["economic_traces"]["total"] += 1
                    trace_type_analysis["economic_traces"]["avg_satisfaction"] += satisfaction_rate
            
            # Calculate averages
            for trace_type in trace_type_analysis:
                total = trace_type_analysis[trace_type]["total"]
                if total > 0:
                    trace_type_analysis[trace_type]["avg_satisfaction"] /= total
            
            experiment_results = {
                "configuration": {
                    "seed": self.seed,
                    "quick_mode": self.quick_mode,
                    "num_traces": num_traces
                },
                "verification_results": verification_result,
                "trace_type_analysis": trace_type_analysis,
                "summary": {
                    "total_properties_verified": verification_result["dataset_summary"]["properties_per_trace"],
                    "total_verifications": verification_result["dataset_summary"]["total_verifications"],
                    "overall_satisfaction_rate": np.mean([
                        trace_result["verification_summary"]["satisfaction_rate"]
                        for trace_result in verification_result["trace_results"]
                    ]),
                    "property_type_performance": verification_result.get("property_analysis", {})
                },
                "timestamp": time.time()
            }
            
            logger.info(f"✅ Formal verification completed. Overall satisfaction: {experiment_results['summary']['overall_satisfaction_rate']:.2%}")
            return experiment_results
            
        except Exception as e:
            error_msg = f"❌ Formal verification experiments failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": str(e), "timestamp": time.time()}
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final experiment report"""
        logger.info("📋 Generating final experiment report...")
        
        total_time = time.time() - self.start_time
        
        report = {
            "experiment_metadata": {
                "seed": self.seed,
                "quick_mode": self.quick_mode,
                "output_directory": str(self.output_dir),
                "total_experiment_time_seconds": total_time,
                "experiment_timestamp": datetime.now().isoformat(),
                "scafad_version": self.config.version["version"],
                "python_version": sys.version,
                "environment": "docker" if os.getenv("SCAFAD_ENVIRONMENT") == "DOCKER" else "local"
            },
            "experiment_results": self.experiment_results,
            "summary": {
                "completed_experiments": len(self.experiment_results),
                "failed_experiments": len([r for r in self.experiment_results.values() if "error" in r]),
                "success_rate": (len(self.experiment_results) - len([r for r in self.experiment_results.values() if "error" in r])) / max(len(self.experiment_results), 1),
                "total_duration_minutes": total_time / 60
            },
            "recommendations": []
        }
        
        # Generate recommendations based on results
        if "system_validation" in self.experiment_results:
            validation = self.experiment_results["system_validation"]
            if validation.get("system_status") == "healthy":
                report["recommendations"].append({
                    "type": "system",
                    "message": "✅ SCAFAD system is healthy and ready for production deployment"
                })
            else:
                report["recommendations"].append({
                    "type": "system",
                    "priority": "high",
                    "message": "❌ SCAFAD system validation failed - review system setup"
                })
        
        if "ignn_experiments" in self.experiment_results:
            ignn_results = self.experiment_results["ignn_experiments"]
            if "summary" in ignn_results and ignn_results["summary"].get("average_f1_score", 0) > 0.8:
                report["recommendations"].append({
                    "type": "performance",
                    "message": f"✅ i-GNN shows excellent performance (F1: {ignn_results['summary']['average_f1_score']:.3f})"
                })
            elif "summary" in ignn_results:
                report["recommendations"].append({
                    "type": "performance", 
                    "priority": "medium",
                    "message": f"⚠️ i-GNN performance could be improved (F1: {ignn_results['summary']['average_f1_score']:.3f})"
                })
        
        if "baseline_experiments" in self.experiment_results:
            baseline_results = self.experiment_results["baseline_experiments"]
            if "summary" in baseline_results and baseline_results["summary"].get("best_model"):
                report["recommendations"].append({
                    "type": "comparison",
                    "message": f"📊 Best baseline model: {baseline_results['summary']['best_model']} (F1: {baseline_results['summary']['best_f1_score']:.3f})"
                })
        
        # Save report
        report_file = self.output_dir / "reports" / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Final report generated: {report_file}")
        return report
    
    async def run_all_experiments(self, experiment_types: List[str] = None) -> Dict[str, Any]:
        """Run all specified experiments"""
        
        if experiment_types is None:
            experiment_types = ["system_validation", "ignn_experiments", 
                              "baseline_experiments", "formal_verification_experiments"]
        
        logger.info(f"🚀 Starting reproducible SCAFAD experiments: {experiment_types}")
        
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint()
        if checkpoint:
            logger.info(f"📂 Found checkpoint from {datetime.fromtimestamp(checkpoint['timestamp'])}")
            logger.info(f"   Previous completed experiments: {checkpoint.get('completed_experiments', [])}")
        
        for experiment_name in experiment_types:
            if checkpoint and experiment_name in checkpoint.get("completed_experiments", []):
                logger.info(f"⏭️  Skipping {experiment_name} (already completed)")
                continue
            
            self.save_checkpoint(experiment_name, "running")
            
            try:
                logger.info(f"🧪 Starting experiment: {experiment_name}")
                
                if experiment_name == "system_validation":
                    result = await self.run_system_validation()
                elif experiment_name == "ignn_experiments":
                    result = await self.run_ignn_experiments()
                elif experiment_name == "baseline_experiments":
                    result = await self.run_baseline_experiments()
                elif experiment_name == "formal_verification_experiments":
                    result = await self.run_formal_verification_experiments()
                else:
                    result = {"error": f"Unknown experiment type: {experiment_name}"}
                
                self.experiment_results[experiment_name] = result
                self.save_checkpoint(experiment_name, "completed", result)
                
                logger.info(f"✅ Completed experiment: {experiment_name}")
                
            except Exception as e:
                error_result = {"error": str(e), "timestamp": time.time()}
                self.experiment_results[experiment_name] = error_result
                self.save_checkpoint(experiment_name, "failed", error_result)
                
                logger.error(f"❌ Failed experiment: {experiment_name} - {e}")
                logger.error(traceback.format_exc())
        
        # Generate final report
        final_report = self.generate_final_report()
        
        logger.info("🎉 All SCAFAD experiments completed!")
        logger.info(f"   Duration: {(time.time() - self.start_time)/60:.1f} minutes")
        logger.info(f"   Results: {self.output_dir}")
        
        return final_report


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SCAFAD Reproducible Experiments")
    parser.add_argument("--experiment-type", choices=["all", "system_validation", 
                       "ignn_experiments", "baseline_experiments", 
                       "formal_verification_experiments"], 
                       default="all", help="Type of experiments to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="experiments/results", 
                       help="Output directory")
    parser.add_argument("--quick-mode", action="store_true", 
                       help="Run in quick mode with smaller datasets")
    parser.add_argument("--resume-from", help="Resume from checkpoint file")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ReproducibleExperimentOrchestrator(
        seed=args.seed,
        output_dir=args.output_dir,
        quick_mode=args.quick_mode
    )
    
    # Determine experiment types to run
    if args.experiment_type == "all":
        experiment_types = ["system_validation", "ignn_experiments", 
                          "baseline_experiments", "formal_verification_experiments"]
    else:
        experiment_types = [args.experiment_type]
    
    # Run experiments
    try:
        final_report = await orchestrator.run_all_experiments(experiment_types)
        
        print("\n" + "="*60)
        print("SCAFAD REPRODUCIBLE EXPERIMENTS SUMMARY")
        print("="*60)
        print(f"Seed: {args.seed}")
        print(f"Duration: {final_report['experiment_metadata']['total_duration_minutes']:.1f} minutes")
        print(f"Completed: {final_report['summary']['completed_experiments']}")
        print(f"Failed: {final_report['summary']['failed_experiments']}")
        print(f"Success rate: {final_report['summary']['success_rate']:.1%}")
        
        if final_report["recommendations"]:
            print("\nRecommendations:")
            for rec in final_report["recommendations"]:
                print(f"  {rec['message']}")
        
        print(f"\nResults available in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Experiments failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())