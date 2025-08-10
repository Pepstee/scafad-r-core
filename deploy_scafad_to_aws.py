#!/usr/bin/env python3
"""
Complete SCAFAD AWS Deployment Script
====================================

This script performs a complete deployment of SCAFAD to AWS Lambda with:
1. Function deployment with optimized configuration
2. Real cold start measurement and analysis
3. Concurrency testing under various loads
4. Performance benchmarking and optimization
5. Cost analysis and monitoring setup
6. Production readiness validation

Usage:
    python deploy_scafad_to_aws.py --function-name scafad-prod --region us-east-1 --test-cold-starts --test-concurrency

Requirements:
    - AWS credentials configured (aws configure or IAM role)
    - Appropriate IAM permissions for Lambda, CloudWatch, IAM
"""

import argparse
import asyncio
import json
import logging
import time
from typing import Dict, List, Any
from pathlib import Path

# Import our deployment system
from aws_deployment.lambda_deployer import (
    AWSLambdaDeployer, LambdaDeploymentConfig, 
    ColdStartMeasurement, ConcurrencyTestResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SCAFADDeploymentOrchestrator:
    """Complete SCAFAD deployment orchestrator"""
    
    def __init__(self, aws_region: str = "us-east-1", profile_name: str = None):
        self.deployer = AWSLambdaDeployer(aws_region, profile_name)
        self.deployment_results = {}
        self.test_results = {}
        
    async def deploy_complete_scafad_system(self, function_name: str, 
                                          memory_sizes: List[int] = None,
                                          test_cold_starts: bool = True,
                                          test_concurrency: bool = True,
                                          test_performance: bool = True) -> Dict[str, Any]:
        """Deploy complete SCAFAD system with comprehensive testing"""
        
        if memory_sizes is None:
            memory_sizes = [256, 512, 1024]  # Test different memory configurations
        
        logger.info("🚀 Starting complete SCAFAD deployment to AWS Lambda")
        logger.info(f"Function name: {function_name}")
        logger.info(f"Memory configurations: {memory_sizes}MB")
        logger.info(f"Testing: cold_starts={test_cold_starts}, concurrency={test_concurrency}, performance={test_performance}")
        
        deployment_start_time = time.time()
        all_results = {}
        
        # Deploy function with different memory configurations
        for memory_size in memory_sizes:
            config_name = f"{function_name}-{memory_size}mb"
            logger.info(f"\n📦 Deploying configuration: {config_name}")
            
            try:
                # Create deployment configuration
                config = LambdaDeploymentConfig(
                    function_name=config_name,
                    runtime="python3.9",
                    memory_size=memory_size,
                    timeout=60,  # Longer timeout for SCAFAD processing
                    environment_variables={
                        "SCAFAD_ENVIRONMENT": "PRODUCTION",
                        "SCAFAD_VERBOSITY": "NORMAL",
                        "SCAFAD_ENABLE_GRAPH": "true",
                        "SCAFAD_ENABLE_ECONOMIC": "true",
                        "SCAFAD_ENABLE_IGNN": "true",
                        "SCAFAD_MAX_MEMORY": str(memory_size),
                        "PYTHONPATH": "/opt/python:/var/runtime:/var/task"
                    },
                    tracing_config={"Mode": "Active"}
                )
                
                # Deploy function
                deployment_info = self.deployer.deploy_scafad_lambda(config, "/workspace")
                logger.info(f"✅ Deployed {config_name}: {deployment_info['function_arn']}")
                
                # Setup monitoring
                monitoring_config = self.deployer.setup_cloudwatch_monitoring(config_name)
                logger.info(f"✅ Monitoring configured for {config_name}")
                
                # Initialize test results
                config_results = {
                    "deployment_info": deployment_info,
                    "monitoring_config": monitoring_config,
                    "memory_size": memory_size,
                    "tests": {}
                }
                
                # Test basic functionality first
                logger.info(f"🧪 Testing basic functionality for {config_name}")
                basic_test_result = await self._test_basic_functionality(config_name)
                config_results["tests"]["basic_functionality"] = basic_test_result
                
                if basic_test_result.get("success", False):
                    logger.info(f"✅ Basic functionality test passed for {config_name}")
                    
                    # Cold start testing
                    if test_cold_starts:
                        logger.info(f"❄️  Testing cold starts for {config_name}")
                        cold_start_measurements = await self.deployer.measure_cold_starts(
                            config_name, num_measurements=20, delay_between_calls=300
                        )
                        config_results["tests"]["cold_starts"] = {
                            "measurements": cold_start_measurements,
                            "analysis": self._analyze_cold_start_measurements(cold_start_measurements)
                        }
                        logger.info(f"✅ Cold start testing completed for {config_name}")
                    
                    # Concurrency testing
                    if test_concurrency:
                        logger.info(f"⚡ Testing concurrency for {config_name}")
                        concurrency_results = await self.deployer.test_concurrency_limits(
                            config_name, target_concurrencies=[1, 5, 10, 20, 50, 100]
                        )
                        config_results["tests"]["concurrency"] = {
                            "results": concurrency_results,
                            "analysis": self._analyze_concurrency_results(concurrency_results)
                        }
                        logger.info(f"✅ Concurrency testing completed for {config_name}")
                    
                    # Performance testing
                    if test_performance:
                        logger.info(f"📊 Running performance benchmarks for {config_name}")
                        performance_results = await self._run_performance_benchmarks(config_name)
                        config_results["tests"]["performance"] = performance_results
                        logger.info(f"✅ Performance testing completed for {config_name}")
                
                else:
                    logger.error(f"❌ Basic functionality test failed for {config_name}, skipping other tests")
                
                all_results[config_name] = config_results
                
            except Exception as e:
                logger.error(f"❌ Failed to deploy/test {config_name}: {e}")
                all_results[config_name] = {
                    "error": str(e),
                    "memory_size": memory_size,
                    "deployment_failed": True
                }
        
        # Generate comprehensive report
        deployment_time = time.time() - deployment_start_time
        
        report = self._generate_deployment_report(all_results, deployment_time)
        
        # Save report
        report_file = f"scafad_aws_deployment_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n🎉 SCAFAD deployment completed in {deployment_time/60:.1f} minutes")
        logger.info(f"📄 Deployment report saved: {report_file}")
        
        return report
    
    async def _test_basic_functionality(self, function_name: str) -> Dict[str, Any]:
        """Test basic SCAFAD functionality"""
        
        try:
            # Test self-test invocation
            test_payload = {
                "test_mode": True,
                "test_type": "basic",
                "timestamp": time.time()
            }
            
            result = await self.deployer.invoke_lambda_async(function_name, test_payload)
            
            success = (
                result["status_code"] == 200 and
                result.get("payload") is not None and
                result.get("error") is None
            )
            
            return {
                "success": success,
                "invocation_result": result,
                "test_payload": test_payload,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _analyze_cold_start_measurements(self, measurements: List[ColdStartMeasurement]) -> Dict[str, Any]:
        """Analyze cold start measurement results"""
        
        if not measurements:
            return {"error": "No measurements provided"}
        
        cold_starts = [m for m in measurements if m.is_cold_start]
        warm_starts = [m for m in measurements if not m.is_cold_start]
        
        analysis = {
            "total_measurements": len(measurements),
            "cold_start_count": len(cold_starts),
            "warm_start_count": len(warm_starts),
            "cold_start_rate": len(cold_starts) / len(measurements) if measurements else 0
        }
        
        if cold_starts:
            cold_durations = [m.total_duration for m in cold_starts]
            init_durations = [m.init_duration for m in cold_starts if m.init_duration > 0]
            
            analysis["cold_start_analysis"] = {
                "duration_ms": {
                    "avg": sum(cold_durations) / len(cold_durations),
                    "min": min(cold_durations),
                    "max": max(cold_durations),
                    "p50": sorted(cold_durations)[len(cold_durations)//2],
                    "p95": sorted(cold_durations)[int(len(cold_durations)*0.95)]
                },
                "init_duration_ms": {
                    "avg": sum(init_durations) / len(init_durations) if init_durations else 0,
                    "min": min(init_durations) if init_durations else 0,
                    "max": max(init_durations) if init_durations else 0
                }
            }
        
        if warm_starts:
            warm_durations = [m.total_duration for m in warm_starts]
            analysis["warm_start_analysis"] = {
                "duration_ms": {
                    "avg": sum(warm_durations) / len(warm_durations),
                    "min": min(warm_durations),
                    "max": max(warm_durations)
                }
            }
        
        # Performance classification
        if cold_starts:
            avg_cold_duration = analysis["cold_start_analysis"]["duration_ms"]["avg"]
            if avg_cold_duration < 3000:  # Less than 3s
                analysis["cold_start_performance"] = "excellent"
            elif avg_cold_duration < 5000:  # Less than 5s
                analysis["cold_start_performance"] = "good"
            elif avg_cold_duration < 10000:  # Less than 10s
                analysis["cold_start_performance"] = "acceptable"
            else:
                analysis["cold_start_performance"] = "needs_optimization"
        
        return analysis
    
    def _analyze_concurrency_results(self, results: List[ConcurrencyTestResult]) -> Dict[str, Any]:
        """Analyze concurrency test results"""
        
        if not results:
            return {"error": "No concurrency results provided"}
        
        analysis = {
            "max_tested_concurrency": max(r.target_concurrency for r in results),
            "successful_configurations": [],
            "failed_configurations": [],
            "optimal_concurrency": None,
            "cost_analysis": []
        }
        
        for result in results:
            config_analysis = {
                "concurrency": result.target_concurrency,
                "success_rate": result.successful_invocations / result.target_concurrency if result.target_concurrency > 0 else 0,
                "error_rate": result.error_rate,
                "cold_start_rate": result.cold_start_rate,
                "avg_duration_ms": result.avg_duration,
                "p95_duration_ms": result.p95_duration,
                "cost_estimate": result.cost_estimate
            }
            
            if config_analysis["success_rate"] >= 0.95 and config_analysis["error_rate"] <= 0.05:
                analysis["successful_configurations"].append(config_analysis)
            else:
                analysis["failed_configurations"].append(config_analysis)
            
            analysis["cost_analysis"].append({
                "concurrency": result.target_concurrency,
                "cost_per_successful_invocation": result.cost_estimate / max(result.successful_invocations, 1),
                "total_cost": result.cost_estimate
            })
        
        # Find optimal concurrency (best cost-performance ratio)
        if analysis["successful_configurations"]:
            optimal = min(
                analysis["successful_configurations"],
                key=lambda x: x["cost_estimate"] / max(x["success_rate"], 0.01)  # Cost per success rate
            )
            analysis["optimal_concurrency"] = optimal["concurrency"]
        
        return analysis
    
    async def _run_performance_benchmarks(self, function_name: str) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks"""
        
        benchmarks = {}
        
        try:
            # Test different payload sizes
            payload_sizes = [
                ("small", {"test_mode": True, "data": "x" * 100}),
                ("medium", {"test_mode": True, "data": "x" * 1000, "records": [{"id": i} for i in range(10)]}),
                ("large", {"test_mode": True, "data": "x" * 10000, "records": [{"id": i, "data": "y" * 100} for i in range(50)]})
            ]
            
            for size_name, payload in payload_sizes:
                logger.info(f"  Testing {size_name} payload size")
                
                # Run multiple invocations
                results = []
                for i in range(10):
                    start_time = time.time()
                    result = await self.deployer.invoke_lambda_async(function_name, payload)
                    end_time = time.time()
                    
                    results.append({
                        "duration_ms": (end_time - start_time) * 1000,
                        "status_code": result["status_code"],
                        "success": result["status_code"] == 200,
                        "payload_size_bytes": len(json.dumps(payload))
                    })
                
                # Analyze results
                successful_results = [r for r in results if r["success"]]
                if successful_results:
                    durations = [r["duration_ms"] for r in successful_results]
                    benchmarks[size_name] = {
                        "payload_size_bytes": results[0]["payload_size_bytes"],
                        "successful_invocations": len(successful_results),
                        "total_invocations": len(results),
                        "success_rate": len(successful_results) / len(results),
                        "duration_ms": {
                            "avg": sum(durations) / len(durations),
                            "min": min(durations),
                            "max": max(durations),
                            "p95": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
                        }
                    }
            
            # Test sustained load
            logger.info("  Testing sustained load (30 invocations over 60s)")
            sustained_results = []
            
            for i in range(30):
                result = await self.deployer.invoke_lambda_async(
                    function_name, 
                    {"test_mode": True, "sustained_test": True, "iteration": i}
                )
                sustained_results.append(result)
                
                # Wait 2 seconds between invocations
                await asyncio.sleep(2)
            
            successful_sustained = [r for r in sustained_results if r["status_code"] == 200]
            benchmarks["sustained_load"] = {
                "total_invocations": len(sustained_results),
                "successful_invocations": len(successful_sustained),
                "success_rate": len(successful_sustained) / len(sustained_results),
                "duration_minutes": 2.0,  # Approximately 2 minutes
                "avg_invocation_interval_s": 2.0
            }
            
        except Exception as e:
            logger.error(f"Performance benchmarking failed: {e}")
            benchmarks["error"] = str(e)
        
        return benchmarks
    
    def _generate_deployment_report(self, all_results: Dict[str, Any], deployment_time: float) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        
        report = {
            "deployment_summary": {
                "total_configurations": len(all_results),
                "successful_deployments": len([r for r in all_results.values() if not r.get("deployment_failed", False)]),
                "failed_deployments": len([r for r in all_results.values() if r.get("deployment_failed", False)]),
                "total_deployment_time_minutes": deployment_time / 60,
                "timestamp": time.time()
            },
            "configurations": all_results,
            "recommendations": [],
            "cost_analysis": {},
            "performance_summary": {}
        }
        
        # Analyze successful configurations
        successful_configs = [
            (name, config) for name, config in all_results.items() 
            if not config.get("deployment_failed", False)
        ]
        
        if successful_configs:
            # Find best performing configuration
            best_config = None
            best_score = 0
            
            for config_name, config_data in successful_configs:
                score = 0
                
                # Score based on cold start performance
                cold_start_tests = config_data.get("tests", {}).get("cold_starts")
                if cold_start_tests and cold_start_tests.get("analysis"):
                    cold_analysis = cold_start_tests["analysis"]
                    performance = cold_analysis.get("cold_start_performance", "needs_optimization")
                    
                    if performance == "excellent":
                        score += 40
                    elif performance == "good":
                        score += 30
                    elif performance == "acceptable":
                        score += 20
                
                # Score based on concurrency performance
                concurrency_tests = config_data.get("tests", {}).get("concurrency")
                if concurrency_tests and concurrency_tests.get("analysis"):
                    conc_analysis = concurrency_tests["analysis"]
                    successful_configs_count = len(conc_analysis.get("successful_configurations", []))
                    score += min(30, successful_configs_count * 5)  # Up to 30 points
                
                # Score based on basic functionality
                basic_test = config_data.get("tests", {}).get("basic_functionality")
                if basic_test and basic_test.get("success"):
                    score += 30
                
                if score > best_score:
                    best_score = score
                    best_config = (config_name, config_data, score)
            
            if best_config:
                report["best_configuration"] = {
                    "name": best_config[0],
                    "memory_size": best_config[1]["memory_size"],
                    "score": best_config[2],
                    "function_arn": best_config[1].get("deployment_info", {}).get("function_arn")
                }
                
                report["recommendations"].append({
                    "type": "deployment",
                    "priority": "high",
                    "message": f"Recommended configuration: {best_config[0]} with {best_config[1]['memory_size']}MB memory"
                })
        
        # Generate cost analysis
        total_estimated_cost = 0
        for config_name, config_data in successful_configs:
            concurrency_tests = config_data.get("tests", {}).get("concurrency", {})
            if "analysis" in concurrency_tests:
                cost_analysis = concurrency_tests["analysis"].get("cost_analysis", [])
                if cost_analysis:
                    max_cost = max(item["total_cost"] for item in cost_analysis)
                    total_estimated_cost += max_cost
        
        report["cost_analysis"] = {
            "total_estimated_test_cost": total_estimated_cost,
            "monthly_estimate_low_usage": total_estimated_cost * 30,  # Very rough estimate
            "cost_per_configuration": {}
        }
        
        # Generate recommendations
        if report["deployment_summary"]["failed_deployments"] > 0:
            report["recommendations"].append({
                "type": "deployment",
                "priority": "high", 
                "message": f"{report['deployment_summary']['failed_deployments']} configuration(s) failed to deploy. Check logs and permissions."
            })
        
        if deployment_time > 1800:  # More than 30 minutes
            report["recommendations"].append({
                "type": "performance",
                "priority": "medium",
                "message": "Deployment took longer than expected. Consider optimizing deployment process."
            })
        
        return report
    
    async def cleanup_all_resources(self, function_prefix: str):
        """Clean up all deployed resources"""
        logger.info(f"🧹 Cleaning up all resources with prefix: {function_prefix}")
        
        # List all deployed functions with the prefix
        functions_to_clean = [
            name for name in self.deployer.deployed_functions.keys()
            if name.startswith(function_prefix)
        ]
        
        for function_name in functions_to_clean:
            try:
                await self.deployer.cleanup_resources(function_name)
                logger.info(f"✅ Cleaned up {function_name}")
            except Exception as e:
                logger.error(f"❌ Failed to clean up {function_name}: {e}")
        
        logger.info("🧹 Cleanup completed")


async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy SCAFAD to AWS Lambda")
    parser.add_argument("--function-name", required=True, help="Base name for Lambda functions")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--memory-sizes", nargs="+", type=int, default=[256, 512, 1024], 
                       help="Memory sizes to test (MB)")
    parser.add_argument("--test-cold-starts", action="store_true", help="Test cold start performance")
    parser.add_argument("--test-concurrency", action="store_true", help="Test concurrency limits")
    parser.add_argument("--test-performance", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--cleanup", action="store_true", help="Clean up resources after testing")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = SCAFADDeploymentOrchestrator(args.region, args.profile)
    
    try:
        # Deploy and test
        report = await orchestrator.deploy_complete_scafad_system(
            function_name=args.function_name,
            memory_sizes=args.memory_sizes,
            test_cold_starts=args.test_cold_starts,
            test_concurrency=args.test_concurrency,
            test_performance=args.test_performance
        )
        
        # Print summary
        print("\n" + "="*60)
        print("SCAFAD AWS DEPLOYMENT SUMMARY")
        print("="*60)
        
        summary = report["deployment_summary"]
        print(f"Total configurations: {summary['total_configurations']}")
        print(f"Successful deployments: {summary['successful_deployments']}")
        print(f"Failed deployments: {summary['failed_deployments']}")
        print(f"Deployment time: {summary['total_deployment_time_minutes']:.1f} minutes")
        
        if "best_configuration" in report:
            best = report["best_configuration"]
            print(f"\n🏆 Best configuration: {best['name']} ({best['memory_size']}MB)")
            print(f"   Function ARN: {best['function_arn']}")
        
        print(f"\n💰 Estimated test cost: ${report['cost_analysis']['total_estimated_test_cost']:.4f}")
        
        if report["recommendations"]:
            print(f"\n📋 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   {rec['priority'].upper()}: {rec['message']}")
        
        # Cleanup if requested
        if args.cleanup:
            await orchestrator.cleanup_all_resources(args.function_name)
            print("\n🧹 Resources cleaned up")
        else:
            print(f"\n⚠️  Resources NOT cleaned up. Use --cleanup or manually delete functions starting with '{args.function_name}'")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())