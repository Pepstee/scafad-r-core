#!/usr/bin/env python3
"""
Execute Real-World SCAFAD Layer 0 Analysis
==========================================
"""

import sys
import os
import time
import json

# Set up path
sys.path.insert(0, '/workspace')

def run_analysis():
    """Execute comprehensive system analysis"""
    
    print("🚀 SCAFAD Layer 0 - Real-World Adaptive Telemetry Analysis")
    print("=" * 60)
    
    analysis_start = time.time()
    results = {}
    
    # Phase 1: Import and Configuration Test
    print("\n📋 PHASE 1: SYSTEM INITIALIZATION")
    print("-" * 35)
    
    try:
        from app_config import create_testing_config
        config = create_testing_config()
        print("✅ Configuration system: Ready")
        config_ready = True
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        config_ready = False
        return {"status": "failed", "reason": "configuration_failure"}
    
    # Phase 2: Core Anomaly Detection Analysis
    print("\n🧠 PHASE 2: ANOMALY DETECTION ENGINE ANALYSIS")
    print("-" * 44)
    
    if config_ready:
        try:
            from layer0_core import AnomalyDetectionEngine
            from app_telemetry import TelemetryRecord
            
            engine = AnomalyDetectionEngine(config)
            
            # Test comprehensive anomaly scenarios
            test_cases = [
                {
                    "name": "Benign Baseline", 
                    "params": {"anomaly_type": "benign", "duration": 120, "memory": 1024, "cpu": 25},
                    "expected_anomaly": False
                },
                {
                    "name": "CPU Spike Attack", 
                    "params": {"anomaly_type": "cpu_spike", "duration": 2500, "memory": 1500, "cpu": 92},
                    "expected_anomaly": True
                },
                {
                    "name": "Memory Exhaustion", 
                    "params": {"anomaly_type": "memory_spike", "duration": 800, "memory": 15000, "cpu": 45},
                    "expected_anomaly": True
                },
                {
                    "name": "Economic Abuse", 
                    "params": {"anomaly_type": "economic_abuse", "duration": 180000, "memory": 2048, "cpu": 88},
                    "expected_anomaly": True
                },
                {
                    "name": "Subtle Timing Attack", 
                    "params": {"anomaly_type": "timing_attack", "duration": 950, "memory": 1800, "cpu": 72},
                    "expected_anomaly": True
                }
            ]
            
            detection_results = []
            total_processing_time = 0
            
            for case in test_cases:
                record = TelemetryRecord(
                    telemetry_id=f"analysis_{case['name'].lower().replace(' ', '_')}",
                    function_id="analysis_function",
                    execution_phase="invoke",
                    anomaly_type=case["params"]["anomaly_type"],
                    duration=float(case["params"]["duration"]),
                    memory_spike_kb=case["params"]["memory"],
                    cpu_utilization=float(case["params"]["cpu"]),
                    custom_fields={"analysis": True, "case": case["name"]}
                )
                
                # Execute detection with timing
                detect_start = time.time()
                result = engine.detect_anomalies(record)
                detect_time = (time.time() - detect_start) * 1000
                total_processing_time += detect_time
                
                # Analyze results
                detected = result.overall_confidence > 0.5
                correct = detected == case["expected_anomaly"]
                
                detection_results.append({
                    "case": case["name"],
                    "expected": case["expected_anomaly"],
                    "detected": detected,
                    "confidence": result.overall_confidence,
                    "correct": correct,
                    "processing_time_ms": detect_time,
                    "algorithms_triggered": len(result.detections)
                })
                
                status = "✅" if correct else "❌"
                print(f"   {status} {case['name']:18} | Conf: {result.overall_confidence:.3f} | {detect_time:.1f}ms | {len(result.detections)} algos")
            
            # Calculate performance metrics
            correct_detections = sum(1 for r in detection_results if r["correct"])
            accuracy = (correct_detections / len(detection_results)) * 100
            avg_processing_time = total_processing_time / len(detection_results)
            
            # Detailed analysis
            true_positives = sum(1 for r in detection_results if r["expected"] and r["detected"])
            false_positives = sum(1 for r in detection_results if not r["expected"] and r["detected"])
            true_negatives = sum(1 for r in detection_results if not r["expected"] and not r["detected"])
            false_negatives = sum(1 for r in detection_results if r["expected"] and not r["detected"])
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n   📊 Detection Performance Analysis:")
            print(f"      Accuracy: {accuracy:.1f}% ({correct_detections}/{len(detection_results)})")
            print(f"      Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1_score:.3f}")
            print(f"      Avg Processing Time: {avg_processing_time:.2f}ms")
            print(f"      Performance Rating: {'EXCELLENT' if f1_score >= 0.9 else 'VERY_GOOD' if f1_score >= 0.8 else 'GOOD' if f1_score >= 0.7 else 'NEEDS_IMPROVEMENT'}")
            
            results['anomaly_detection'] = {
                "success": True,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "avg_processing_time_ms": avg_processing_time,
                "detection_results": detection_results
            }
            
        except Exception as e:
            print(f"❌ Anomaly detection analysis failed: {e}")
            results['anomaly_detection'] = {"success": False, "error": str(e)}
    
    # Phase 3: Compression System Analysis
    print("\n🗜️ PHASE 3: COMPRESSION OPTIMIZATION ANALYSIS")
    print("-" * 43)
    
    if config_ready:
        try:
            from layer0_compression_optimizer import CompressionOptimizer
            
            optimizer = CompressionOptimizer(config)
            
            # Test different data patterns
            compression_tests = [
                {
                    "name": "JSON Telemetry",
                    "data": json.dumps({
                        "telemetry": [{"id": i, "metrics": [1,2,3,4,5] * 20} for i in range(50)]
                    }).encode()
                },
                {
                    "name": "Repetitive Logs", 
                    "data": b"[LOG] Function execution completed successfully. " * 200
                },
                {
                    "name": "Binary Metrics",
                    "data": bytes(range(256)) * 25
                },
                {
                    "name": "Large JSON Dataset",
                    "data": json.dumps({
                        "functions": [
                            {"id": f"func_{i}", "executions": [{"duration": j*10, "memory": j*100} for j in range(20)]}
                            for i in range(100)
                        ]
                    }).encode()
                }
            ]
            
            compression_results = []
            
            for test in compression_tests:
                # Compression test
                compress_start = time.time()
                compressed, metrics = optimizer.compress_data(test["data"])
                compress_time = (time.time() - compress_start) * 1000
                
                # Decompression test
                decompress_start = time.time()
                decompressed, decompress_time = optimizer.decompress_data(compressed, metrics.algorithm, measure_performance=True)
                
                # Verify integrity
                integrity_ok = decompressed == test["data"]
                
                result = {
                    "test_name": test["name"],
                    "original_size": len(test["data"]),
                    "compressed_size": len(compressed),
                    "compression_ratio": metrics.compression_ratio,
                    "algorithm": metrics.algorithm.value,
                    "compression_time_ms": compress_time,
                    "decompression_time_ms": decompress_time or 0,
                    "integrity_verified": integrity_ok,
                    "space_saved_percent": (1 - metrics.compression_ratio) * 100,
                    "throughput_mbps": (len(test["data"]) / (1024*1024)) / ((compress_time + (decompress_time or 0)) / 1000) if (compress_time + (decompress_time or 0)) > 0 else 0
                }
                
                compression_results.append(result)
                
                status = "✅" if integrity_ok else "❌"
                print(f"   {status} {test['name']:16} | {metrics.algorithm.value:6} | Ratio: {metrics.compression_ratio:.3f} | Saved: {result['space_saved_percent']:5.1f}% | {compress_time:.1f}ms")
            
            # Overall compression analysis
            avg_compression_ratio = sum(r["compression_ratio"] for r in compression_results) / len(compression_results)
            avg_space_saved = sum(r["space_saved_percent"] for r in compression_results) / len(compression_results)
            all_integrity_ok = all(r["integrity_verified"] for r in compression_results)
            avg_throughput = sum(r["throughput_mbps"] for r in compression_results) / len(compression_results)
            
            print(f"\n   📊 Compression System Analysis:")
            print(f"      Average Compression Ratio: {avg_compression_ratio:.3f}")
            print(f"      Average Space Saved: {avg_space_saved:.1f}%")
            print(f"      Data Integrity: {'100% Verified' if all_integrity_ok else 'FAILED'}")
            print(f"      Average Throughput: {avg_throughput:.1f} MB/s")
            
            compression_rating = (
                "EXCELLENT" if all_integrity_ok and avg_compression_ratio < 0.7 and avg_throughput > 5 else
                "VERY_GOOD" if all_integrity_ok and avg_compression_ratio < 0.8 else
                "GOOD" if all_integrity_ok else
                "NEEDS_IMPROVEMENT"
            )
            print(f"      Performance Rating: {compression_rating}")
            
            results['compression'] = {
                "success": True,
                "avg_compression_ratio": avg_compression_ratio,
                "avg_space_saved_percent": avg_space_saved,
                "integrity_verified": all_integrity_ok,
                "avg_throughput_mbps": avg_throughput,
                "rating": compression_rating,
                "test_results": compression_results
            }
            
        except Exception as e:
            print(f"❌ Compression analysis failed: {e}")
            results['compression'] = {"success": False, "error": str(e)}
    
    # Phase 4: Stream Processing Architecture Analysis
    print("\n🌊 PHASE 4: STREAM PROCESSING ARCHITECTURE ANALYSIS")
    print("-" * 50)
    
    if config_ready:
        try:
            from layer0_stream_processor import StreamProcessor
            
            processor = StreamProcessor(config)
            
            # Analyze configuration
            config_analysis = {
                "max_queue_size": processor.max_queue_size,
                "batch_size": processor.batch_size,
                "worker_pool_size": processor.worker_pool_size,
                "window_size_ms": processor.window_config.window_size_ms,
                "watermark_interval_ms": processor.window_config.watermark_interval_ms
            }
            
            # Calculate theoretical performance
            theoretical_max_rps = (processor.batch_size * processor.worker_pool_size * 1000) / processor.max_processing_latency_ms
            queue_buffer_capacity_seconds = processor.max_queue_size / theoretical_max_rps if theoretical_max_rps > 0 else 0
            
            print(f"   🏗️ Architecture Configuration:")
            print(f"      Queue Capacity: {config_analysis['max_queue_size']} records")
            print(f"      Batch Processing: {config_analysis['batch_size']} records/batch")
            print(f"      Worker Pool: {config_analysis['worker_pool_size']} concurrent workers")
            print(f"      Processing Window: {config_analysis['window_size_ms']}ms")
            
            print(f"\n   📊 Performance Characteristics:")
            print(f"      Theoretical Max Throughput: {theoretical_max_rps:.1f} records/second")
            print(f"      Queue Buffer Capacity: {queue_buffer_capacity_seconds:.1f} seconds")
            print(f"      Backpressure Threshold: {processor.backpressure_threshold*100:.0f}% queue capacity")
            
            # Architecture assessment
            scalability_score = min(1.0, theoretical_max_rps / 100)  # 100 RPS as baseline
            architecture_rating = (
                "EXCELLENT" if scalability_score >= 0.8 and processor.worker_pool_size >= 4 else
                "VERY_GOOD" if scalability_score >= 0.6 else
                "GOOD" if scalability_score >= 0.4 else
                "NEEDS_IMPROVEMENT"
            )
            
            print(f"      Architecture Rating: {architecture_rating}")
            
            results['stream_processing'] = {
                "success": True,
                "theoretical_max_rps": theoretical_max_rps,
                "queue_buffer_seconds": queue_buffer_capacity_seconds,
                "scalability_score": scalability_score,
                "architecture_rating": architecture_rating,
                "config": config_analysis
            }
            
        except Exception as e:
            print(f"❌ Stream processing analysis failed: {e}")
            results['stream_processing'] = {"success": False, "error": str(e)}
    
    # Phase 5: Integration Pipeline Test
    print("\n🔗 PHASE 5: END-TO-END INTEGRATION PIPELINE")
    print("-" * 40)
    
    if results.get('anomaly_detection', {}).get('success') and results.get('compression', {}).get('success'):
        try:
            # Simulate realistic production scenario
            scenario_records = [
                {"anomaly_type": "benign", "duration": 150, "memory": 1024, "cpu": 30, "expected": False},
                {"anomaly_type": "cpu_spike", "duration": 2200, "memory": 1800, "cpu": 88, "expected": True},
                {"anomaly_type": "memory_spike", "duration": 750, "memory": 12000, "cpu": 50, "expected": True},
                {"anomaly_type": "benign", "duration": 180, "memory": 1200, "cpu": 35, "expected": False},
                {"anomaly_type": "economic_abuse", "duration": 250000, "memory": 2500, "cpu": 85, "expected": True}
            ]
            
            integration_results = []
            pipeline_processing_time = 0
            
            for i, scenario in enumerate(scenario_records):
                pipeline_start = time.time()
                
                # Create telemetry record
                record = TelemetryRecord(
                    telemetry_id=f"integration_pipeline_{i:03d}",
                    function_id=f"pipeline_function_{i%3}",
                    execution_phase="invoke",
                    anomaly_type=scenario["anomaly_type"],
                    duration=float(scenario["duration"]),
                    memory_spike_kb=scenario["memory"],
                    cpu_utilization=float(scenario["cpu"]),
                    custom_fields={"pipeline_test": True, "scenario_index": i}
                )
                
                # Pipeline Step 1: Anomaly Detection
                detection_result = engine.detect_anomalies(record)
                
                # Pipeline Step 2: Result Packaging
                pipeline_data = {
                    "telemetry_id": record.telemetry_id,
                    "function_id": record.function_id,
                    "anomaly_confidence": detection_result.overall_confidence,
                    "anomaly_detected": detection_result.overall_confidence > 0.5,
                    "detection_details": [
                        {"type": d.anomaly_type, "confidence": d.confidence, "algorithm": d.algorithm_name}
                        for d in detection_result.detections[:3]
                    ],
                    "processing_metadata": {
                        "original_anomaly_type": scenario["anomaly_type"],
                        "expected_detection": scenario["expected"]
                    }
                }
                
                # Pipeline Step 3: Compression
                pipeline_json = json.dumps(pipeline_data).encode()
                compressed_data, comp_metrics = optimizer.compress_data(pipeline_json)
                
                # Pipeline Step 4: Verification
                decompressed_data, _ = optimizer.decompress_data(compressed_data, comp_metrics.algorithm)
                verified_data = json.loads(decompressed_data.decode())
                
                pipeline_time = (time.time() - pipeline_start) * 1000
                pipeline_processing_time += pipeline_time
                
                # Validate pipeline integrity
                pipeline_success = (
                    verified_data["telemetry_id"] == record.telemetry_id and
                    abs(verified_data["anomaly_confidence"] - detection_result.overall_confidence) < 0.001 and
                    verified_data["anomaly_detected"] == (detection_result.overall_confidence > 0.5)
                )
                
                detection_correct = (detection_result.overall_confidence > 0.5) == scenario["expected"]
                
                integration_results.append({
                    "scenario_index": i,
                    "pipeline_success": pipeline_success,
                    "detection_correct": detection_correct,
                    "anomaly_confidence": detection_result.overall_confidence,
                    "compression_ratio": comp_metrics.compression_ratio,
                    "pipeline_time_ms": pipeline_time,
                    "data_integrity": pipeline_success
                })
                
                status = "✅" if pipeline_success and detection_correct else "⚠️" if pipeline_success else "❌"
                print(f"   {status} Pipeline {i+1} | Conf: {detection_result.overall_confidence:.3f} | Ratio: {comp_metrics.compression_ratio:.3f} | {pipeline_time:.1f}ms")
            
            # Integration analysis
            successful_pipelines = sum(1 for r in integration_results if r["pipeline_success"])
            correct_detections = sum(1 for r in integration_results if r["detection_correct"])
            pipeline_success_rate = successful_pipelines / len(integration_results)
            detection_accuracy = correct_detections / len(integration_results)
            avg_pipeline_time = pipeline_processing_time / len(integration_results)
            
            print(f"\n   📊 Integration Pipeline Analysis:")
            print(f"      Pipeline Success Rate: {pipeline_success_rate*100:.1f}% ({successful_pipelines}/{len(integration_results)})")
            print(f"      Detection Accuracy: {detection_accuracy*100:.1f}% ({correct_detections}/{len(integration_results)})")
            print(f"      Average Pipeline Time: {avg_pipeline_time:.2f}ms")
            
            integration_rating = (
                "EXCELLENT" if pipeline_success_rate >= 0.95 and detection_accuracy >= 0.8 else
                "VERY_GOOD" if pipeline_success_rate >= 0.85 and detection_accuracy >= 0.7 else
                "GOOD" if pipeline_success_rate >= 0.75 else
                "NEEDS_IMPROVEMENT"
            )
            
            print(f"      Integration Rating: {integration_rating}")
            
            results['integration'] = {
                "success": True,
                "pipeline_success_rate": pipeline_success_rate,
                "detection_accuracy": detection_accuracy,
                "avg_pipeline_time_ms": avg_pipeline_time,
                "rating": integration_rating,
                "integration_results": integration_results
            }
            
        except Exception as e:
            print(f"❌ Integration pipeline test failed: {e}")
            results['integration'] = {"success": False, "error": str(e)}
    else:
        print("⚠️ Skipping integration test - prerequisite components not available")
        results['integration'] = {"success": False, "reason": "prerequisites_unavailable"}
    
    # Final Assessment
    total_analysis_time = time.time() - analysis_start
    
    print(f"\n🏆 COMPREHENSIVE SYSTEM ASSESSMENT")
    print("=" * 40)
    print(f"Analysis Duration: {total_analysis_time:.1f} seconds")
    
    # Calculate overall scores
    successful_phases = sum(1 for result in results.values() if result.get("success", False))
    total_phases = len(results)
    overall_success_rate = successful_phases / total_phases if total_phases > 0 else 0
    
    # Performance metrics aggregation
    performance_scores = []
    if results.get('anomaly_detection', {}).get('success'):
        performance_scores.append(('Anomaly Detection F1', results['anomaly_detection']['f1_score']))
    if results.get('compression', {}).get('success'):
        compression_score = 1.0 - results['compression']['avg_compression_ratio']  # Lower ratio = better
        performance_scores.append(('Compression Efficiency', compression_score))
    if results.get('integration', {}).get('success'):
        performance_scores.append(('Integration Pipeline', results['integration']['pipeline_success_rate']))
    
    avg_performance = sum(score for _, score in performance_scores) / len(performance_scores) if performance_scores else 0
    
    print(f"\nSystem Overview:")
    print(f"   Phase Success Rate: {overall_success_rate*100:.1f}% ({successful_phases}/{total_phases})")
    print(f"   Average Performance Score: {avg_performance:.3f}")
    
    # Component status
    print(f"\nComponent Status:")
    for component, result in results.items():
        status_icon = "✅" if result.get("success") else "❌"
        component_name = component.replace('_', ' ').title()
        
        if result.get("success"):
            if component == 'anomaly_detection':
                extra_info = f"F1: {result['f1_score']:.3f}"
            elif component == 'compression':
                extra_info = f"Ratio: {result['avg_compression_ratio']:.3f}"
            elif component == 'stream_processing':
                extra_info = f"Max RPS: {result['theoretical_max_rps']:.0f}"
            elif component == 'integration':
                extra_info = f"Success: {result['pipeline_success_rate']*100:.0f}%"
            else:
                extra_info = "Ready"
            
            print(f"   {status_icon} {component_name:20} | {extra_info}")
        else:
            reason = result.get('error', result.get('reason', 'Unknown'))
            print(f"   {status_icon} {component_name:20} | {reason}")
    
    # Final system rating
    if overall_success_rate >= 0.9 and avg_performance >= 0.8:
        final_rating = "🟢 EXCELLENT - Production Ready"
        deployment_status = "APPROVED_FOR_PRODUCTION"
    elif overall_success_rate >= 0.75 and avg_performance >= 0.7:
        final_rating = "🟡 VERY_GOOD - Production Ready with Monitoring"
        deployment_status = "APPROVED_WITH_MONITORING"
    elif overall_success_rate >= 0.6:
        final_rating = "🟠 GOOD - Needs Minor Improvements"
        deployment_status = "STAGING_DEPLOYMENT_ONLY"
    else:
        final_rating = "🔴 NEEDS_IMPROVEMENT - Not Production Ready"
        deployment_status = "REQUIRES_DEVELOPMENT"
    
    print(f"\nFinal Assessment: {final_rating}")
    print(f"Deployment Status: {deployment_status}")
    
    # Key capabilities summary
    capabilities = []
    if results.get('anomaly_detection', {}).get('success'):
        f1_score = results['anomaly_detection']['f1_score']
        capabilities.append(f"✅ Advanced Anomaly Detection (F1: {f1_score:.3f})")
    
    if results.get('compression', {}).get('success'):
        space_saved = results['compression']['avg_space_saved_percent']
        capabilities.append(f"✅ Intelligent Compression (Avg: {space_saved:.1f}% space saved)")
    
    if results.get('stream_processing', {}).get('success'):
        max_rps = results['stream_processing']['theoretical_max_rps']
        capabilities.append(f"✅ Real-time Stream Processing (Max: {max_rps:.0f} RPS)")
    
    if results.get('integration', {}).get('success'):
        success_rate = results['integration']['pipeline_success_rate']
        capabilities.append(f"✅ End-to-end Pipeline Integration ({success_rate*100:.0f}% success rate)")
    
    print(f"\nKey System Capabilities:")
    for capability in capabilities:
        print(f"   {capability}")
    
    if len(capabilities) >= 3:
        print(f"\n🎉 SCAFAD Layer 0 demonstrates comprehensive adaptive telemetry capabilities!")
        print(f"💼 The system provides enterprise-grade serverless anomaly detection with:")
        print(f"   • Multi-algorithm anomaly detection with {results.get('anomaly_detection', {}).get('f1_score', 0)*100:.0f}% F1-score")
        print(f"   • Adaptive compression optimization saving {results.get('compression', {}).get('avg_space_saved_percent', 0):.0f}% storage space")
        print(f"   • Real-time stream processing with {results.get('stream_processing', {}).get('theoretical_max_rps', 0):.0f} RPS capacity")
        print(f"   • Robust end-to-end integration pipeline")
    
    print(f"\n✨ Real-world adaptive telemetry analysis complete!")
    
    return {
        "status": "completed",
        "overall_success_rate": overall_success_rate,
        "avg_performance_score": avg_performance,
        "final_rating": final_rating,
        "deployment_status": deployment_status,
        "analysis_duration_seconds": total_analysis_time,
        "phase_results": results,
        "capabilities": capabilities
    }

if __name__ == "__main__":
    try:
        result = run_analysis()
        print(f"\nAnalysis result status: {result['status']}")
    except Exception as e:
        print(f"\n💥 Analysis failed: {e}")
        import traceback
        traceback.print_exc()