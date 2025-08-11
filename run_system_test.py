#!/usr/bin/env python3
"""
SCAFAD Layer 0: Simplified Real-World System Test Runner
========================================================

A focused test runner that works with available components to simulate
real-world adaptive telemetry scenarios and analyze system performance.
"""

import sys
import os
import time
import json
import asyncio
import random
from typing import Dict, List, Any

# Add workspace to Python path
sys.path.insert(0, '/workspace')

def test_system_imports():
    """Test if core system components can be imported"""
    print("🔍 Testing system component imports...")
    
    import_results = {}
    
    # Test core imports
    try:
        from app_config import create_testing_config, get_default_config
        import_results['app_config'] = True
        print("   ✅ app_config imported successfully")
    except ImportError as e:
        import_results['app_config'] = False
        print(f"   ❌ app_config import failed: {e}")
    
    try:
        from app_telemetry import TelemetryRecord, MultiChannelTelemetry
        import_results['app_telemetry'] = True
        print("   ✅ app_telemetry imported successfully")
    except ImportError as e:
        import_results['app_telemetry'] = False
        print(f"   ❌ app_telemetry import failed: {e}")
    
    try:
        from layer0_core import AnomalyDetectionEngine
        import_results['layer0_core'] = True
        print("   ✅ layer0_core imported successfully")
    except ImportError as e:
        import_results['layer0_core'] = False
        print(f"   ❌ layer0_core import failed: {e}")
    
    try:
        from layer0_stream_processor import StreamProcessor
        import_results['layer0_stream_processor'] = True
        print("   ✅ layer0_stream_processor imported successfully")
    except ImportError as e:
        import_results['layer0_stream_processor'] = False
        print(f"   ❌ layer0_stream_processor import failed: {e}")
    
    try:
        from layer0_compression_optimizer import CompressionOptimizer
        import_results['layer0_compression_optimizer'] = True
        print("   ✅ layer0_compression_optimizer imported successfully")
    except ImportError as e:
        import_results['layer0_compression_optimizer'] = False
        print(f"   ❌ layer0_compression_optimizer import failed: {e}")
    
    try:
        from utils.test_data_generator import generate_test_payloads
        import_results['test_data_generator'] = True
        print("   ✅ test_data_generator imported successfully")
    except ImportError as e:
        import_results['test_data_generator'] = False
        print(f"   ❌ test_data_generator import failed: {e}")
    
    try:
        from app_main import Layer0_AdaptiveTelemetryController
        import_results['app_main'] = True
        print("   ✅ app_main imported successfully")
    except ImportError as e:
        import_results['app_main'] = False
        print(f"   ❌ app_main import failed: {e}")
    
    success_count = sum(1 for success in import_results.values() if success)
    total_count = len(import_results)
    
    print(f"\n📊 Import Results: {success_count}/{total_count} successful ({success_count/total_count*100:.1f}%)")
    
    return import_results

def test_anomaly_detection_engine():
    """Test the anomaly detection engine with various scenarios"""
    print("\n🧠 Testing Anomaly Detection Engine...")
    
    try:
        from app_config import create_testing_config
        from layer0_core import AnomalyDetectionEngine
        from app_telemetry import TelemetryRecord
        
        config = create_testing_config()
        engine = AnomalyDetectionEngine(config)
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Benign Execution",
                "telemetry": TelemetryRecord(
                    telemetry_id="test_benign_001",
                    function_id="test-function",
                    execution_phase="invoke",
                    anomaly_type="benign",
                    duration=120.5,
                    memory_spike_kb=1024,
                    cpu_utilization=25.0,
                    custom_fields={"test_case": "benign"}
                ),
                "expected_anomaly": False
            },
            {
                "name": "CPU Spike Anomaly",
                "telemetry": TelemetryRecord(
                    telemetry_id="test_cpu_spike_001",
                    function_id="test-function",
                    execution_phase="invoke",
                    anomaly_type="cpu_spike",
                    duration=2500.0,
                    memory_spike_kb=1024,
                    cpu_utilization=95.0,
                    custom_fields={"test_case": "cpu_spike"}
                ),
                "expected_anomaly": True
            },
            {
                "name": "Memory Spike Anomaly",
                "telemetry": TelemetryRecord(
                    telemetry_id="test_memory_spike_001",
                    function_id="test-function",
                    execution_phase="invoke",
                    anomaly_type="memory_spike",
                    duration=800.0,
                    memory_spike_kb=15000,
                    cpu_utilization=45.0,
                    custom_fields={"test_case": "memory_spike"}
                ),
                "expected_anomaly": True
            },
            {
                "name": "Economic Abuse Pattern",
                "telemetry": TelemetryRecord(
                    telemetry_id="test_economic_abuse_001",
                    function_id="test-function",
                    execution_phase="invoke",
                    anomaly_type="economic_abuse",
                    duration=250000.0,  # Very long execution
                    memory_spike_kb=2048,
                    cpu_utilization=88.0,
                    custom_fields={"test_case": "economic_abuse"}
                ),
                "expected_anomaly": True
            }
        ]
        
        detection_results = []
        
        for scenario in test_scenarios:
            start_time = time.time()
            result = engine.detect_anomalies(scenario["telemetry"])
            processing_time = (time.time() - start_time) * 1000
            
            is_anomaly_detected = result.overall_confidence > 0.5
            correct_detection = is_anomaly_detected == scenario["expected_anomaly"]
            
            detection_results.append({
                "scenario": scenario["name"],
                "expected_anomaly": scenario["expected_anomaly"],
                "detected_anomaly": is_anomaly_detected,
                "confidence": result.overall_confidence,
                "processing_time_ms": processing_time,
                "correct_detection": correct_detection,
                "num_detections": len(result.detections)
            })
            
            status_icon = "✅" if correct_detection else "❌"
            print(f"   {status_icon} {scenario['name']}: confidence={result.overall_confidence:.3f}, time={processing_time:.2f}ms")
        
        # Calculate accuracy
        correct_detections = sum(1 for r in detection_results if r["correct_detection"])
        accuracy = correct_detections / len(detection_results) * 100
        avg_processing_time = sum(r["processing_time_ms"] for r in detection_results) / len(detection_results)
        
        print(f"\n📊 Detection Accuracy: {accuracy:.1f}% ({correct_detections}/{len(detection_results)})")
        print(f"📊 Average Processing Time: {avg_processing_time:.2f}ms")
        
        return {
            "success": accuracy >= 75.0,
            "accuracy": accuracy,
            "avg_processing_time_ms": avg_processing_time,
            "results": detection_results
        }
        
    except Exception as e:
        print(f"❌ Anomaly detection engine test failed: {e}")
        return {"success": False, "error": str(e)}

def test_compression_optimizer():
    """Test the compression optimization system"""
    print("\n🗜️ Testing Compression Optimizer...")
    
    try:
        from app_config import create_testing_config
        from layer0_compression_optimizer import CompressionOptimizer
        
        config = create_testing_config()
        optimizer = CompressionOptimizer(config)
        
        # Test data scenarios
        test_datasets = [
            {
                "name": "JSON Telemetry",
                "data": json.dumps({
                    "telemetry_id": "test_001",
                    "timestamp": time.time(),
                    "metrics": [1, 2, 3, 4, 5] * 100,
                    "anomaly_scores": [0.1, 0.2, 0.05, 0.8, 0.3] * 50
                }).encode(),
                "type": "json"
            },
            {
                "name": "Repetitive Data",
                "data": b"TELEMETRY_LOG_ENTRY_" * 500,
                "type": "text"
            },
            {
                "name": "Binary Data",
                "data": bytes(range(256)) * 20,
                "type": "binary"
            }
        ]
        
        compression_results = []
        
        for dataset in test_datasets:
            start_time = time.time()
            compressed, metrics = optimizer.compress_data(dataset["data"])
            processing_time = (time.time() - start_time) * 1000
            
            # Test decompression
            decompression_start = time.time()
            decompressed, decompression_time = optimizer.decompress_data(
                compressed, metrics.algorithm, measure_performance=True
            )
            
            decompression_success = decompressed == dataset["data"]
            
            compression_results.append({
                "dataset": dataset["name"],
                "original_size": len(dataset["data"]),
                "compressed_size": len(compressed),
                "compression_ratio": metrics.compression_ratio,
                "algorithm": metrics.algorithm.value,
                "compression_time_ms": processing_time,
                "decompression_time_ms": decompression_time or 0.0,
                "decompression_success": decompression_success,
                "throughput_mbps": metrics.throughput_mbps
            })
            
            status_icon = "✅" if decompression_success else "❌"
            print(f"   {status_icon} {dataset['name']}: {metrics.algorithm.value}, "
                  f"ratio={metrics.compression_ratio:.3f}, "
                  f"time={processing_time:.2f}ms")
        
        # Calculate overall performance
        avg_compression_ratio = sum(r["compression_ratio"] for r in compression_results) / len(compression_results)
        all_successful = all(r["decompression_success"] for r in compression_results)
        avg_throughput = sum(r["throughput_mbps"] for r in compression_results) / len(compression_results)
        
        print(f"\n📊 Average Compression Ratio: {avg_compression_ratio:.3f}")
        print(f"📊 Decompression Success Rate: {100.0 if all_successful else 0.0}%")
        print(f"📊 Average Throughput: {avg_throughput:.1f} MB/s")
        
        return {
            "success": all_successful and avg_compression_ratio < 0.9,
            "avg_compression_ratio": avg_compression_ratio,
            "decompression_success_rate": 1.0 if all_successful else 0.0,
            "avg_throughput_mbps": avg_throughput,
            "results": compression_results
        }
        
    except Exception as e:
        print(f"❌ Compression optimizer test failed: {e}")
        return {"success": False, "error": str(e)}

async def test_stream_processor():
    """Test the stream processing pipeline"""
    print("\n🌊 Testing Stream Processor...")
    
    try:
        from app_config import create_testing_config
        from layer0_stream_processor import StreamProcessor
        from app_telemetry import TelemetryRecord
        
        config = create_testing_config()
        processor = StreamProcessor(config)
        
        # Start the processor
        start_success = await processor.start()
        if not start_success:
            return {"success": False, "error": "Failed to start stream processor"}
        
        # Generate test telemetry records
        test_records = []
        for i in range(50):
            record = TelemetryRecord(
                telemetry_id=f"stream_test_{i}",
                function_id=f"test-function-{i}",
                execution_phase="invoke",
                anomaly_type="benign" if i % 5 != 0 else "cpu_spike",
                duration=100.0 + random.uniform(0, 200),
                memory_spike_kb=1024 + random.randint(0, 2048),
                cpu_utilization=25.0 + random.uniform(0, 50),
                custom_fields={"batch_id": "stream_test", "record_index": i}
            )
            test_records.append(record)
        
        # Ingest records into stream processor
        print(f"   Ingesting {len(test_records)} telemetry records...")
        ingestion_start = time.time()
        ingested_count = await processor.ingest_telemetry_batch(test_records)
        ingestion_time = (time.time() - ingestion_start) * 1000
        
        print(f"   Ingested {ingested_count}/{len(test_records)} records in {ingestion_time:.2f}ms")
        
        # Wait for processing results
        print("   Waiting for stream processing results...")
        results_collected = []
        timeout = 15  # 15 seconds timeout
        start_wait = time.time()
        
        while len(results_collected) < 3 and (time.time() - start_wait) < timeout:
            result = await processor.get_processing_results(timeout_ms=2000)
            if result:
                results_collected.append(result)
                print(f"   📊 Window processed: {result.record_count} records, "
                      f"{len(result.anomaly_results)} anomalies detected")
        
        # Get final metrics
        stream_metrics = processor.get_stream_metrics()
        
        # Stop the processor
        await processor.stop()
        
        success = (
            ingested_count >= len(test_records) * 0.8 and  # At least 80% ingested
            len(results_collected) > 0 and  # Got some processing results
            stream_metrics.error_count < 5  # Low error count
        )
        
        print(f"\n📊 Stream Processing Results:")
        print(f"   Records Ingested: {ingested_count}/{len(test_records)} ({ingested_count/len(test_records)*100:.1f}%)")
        print(f"   Processing Results: {len(results_collected)} windows")
        print(f"   Records/Second: {stream_metrics.records_per_second:.1f}")
        print(f"   Processing Latency: {stream_metrics.processing_latency_ms:.2f}ms")
        print(f"   Error Count: {stream_metrics.error_count}")
        
        return {
            "success": success,
            "ingested_records": ingested_count,
            "total_records": len(test_records),
            "ingestion_rate": ingested_count / len(test_records),
            "processing_results": len(results_collected),
            "records_per_second": stream_metrics.records_per_second,
            "processing_latency_ms": stream_metrics.processing_latency_ms,
            "error_count": stream_metrics.error_count
        }
        
    except Exception as e:
        print(f"❌ Stream processor test failed: {e}")
        return {"success": False, "error": str(e)}

def test_telemetry_channels():
    """Test multi-channel telemetry system"""
    print("\n📡 Testing Multi-Channel Telemetry...")
    
    try:
        from app_config import create_testing_config
        from app_telemetry import MultiChannelTelemetry, TelemetryRecord
        
        config = create_testing_config()
        telemetry = MultiChannelTelemetry(config.telemetry)
        
        # Test telemetry emission
        test_records = []
        for i in range(20):
            record = TelemetryRecord(
                telemetry_id=f"telemetry_test_{i}",
                function_id=f"test-function-{i}",
                execution_phase="invoke",
                anomaly_type="benign",
                duration=100.0 + i * 10,
                memory_spike_kb=1024,
                cpu_utilization=25.0,
                custom_fields={"test_batch": "telemetry_channels"}
            )
            test_records.append(record)
        
        # Emit telemetry records
        successful_emissions = 0
        emission_times = []
        
        for record in test_records:
            start_time = time.time()
            try:
                success = await telemetry.emit_telemetry(record)
                emission_time = (time.time() - start_time) * 1000
                emission_times.append(emission_time)
                
                if success:
                    successful_emissions += 1
                    
            except Exception as e:
                print(f"   ⚠️ Emission failed for {record.telemetry_id}: {e}")
        
        # Get channel status
        channel_status = telemetry.get_channel_status()
        
        success_rate = successful_emissions / len(test_records) if test_records else 0
        avg_emission_time = sum(emission_times) / len(emission_times) if emission_times else 0
        
        print(f"\n📊 Telemetry Channel Results:")
        print(f"   Active Channels: {len(channel_status)}")
        print(f"   Successful Emissions: {successful_emissions}/{len(test_records)} ({success_rate*100:.1f}%)")
        print(f"   Average Emission Time: {avg_emission_time:.2f}ms")
        
        # Display channel details
        for channel_name, status in channel_status.items():
            print(f"   📺 {channel_name}: {status}")
        
        return {
            "success": success_rate >= 0.8,  # At least 80% success rate
            "success_rate": success_rate,
            "active_channels": len(channel_status),
            "avg_emission_time_ms": avg_emission_time,
            "channel_status": channel_status
        }
        
    except Exception as e:
        print(f"❌ Telemetry channels test failed: {e}")
        return {"success": False, "error": str(e)}

def test_integration_scenario():
    """Test full system integration with realistic scenario"""
    print("\n🔗 Testing Full System Integration...")
    
    try:
        from app_config import create_testing_config
        from layer0_core import AnomalyDetectionEngine
        from layer0_compression_optimizer import CompressionOptimizer
        from app_telemetry import TelemetryRecord
        
        config = create_testing_config()
        
        # Initialize components
        anomaly_engine = AnomalyDetectionEngine(config)
        compression_optimizer = CompressionOptimizer(config)
        
        # Simulate realistic serverless execution scenario
        print("   Simulating serverless execution scenario...")
        
        # Generate realistic telemetry data
        scenario_records = []
        
        # Normal executions (70%)
        for i in range(35):
            record = TelemetryRecord(
                telemetry_id=f"integration_normal_{i}",
                function_id=f"lambda-function-{i%5}",  # 5 different functions
                execution_phase="invoke",
                anomaly_type="benign",
                duration=random.uniform(50, 300),
                memory_spike_kb=random.randint(512, 2048),
                cpu_utilization=random.uniform(10, 40),
                custom_fields={
                    "integration_test": True,
                    "scenario": "normal_execution"
                }
            )
            scenario_records.append(record)
        
        # CPU spike anomalies (20%)
        for i in range(10):
            record = TelemetryRecord(
                telemetry_id=f"integration_cpu_spike_{i}",
                function_id=f"lambda-function-{i%3}",
                execution_phase="invoke",
                anomaly_type="cpu_spike",
                duration=random.uniform(800, 2500),
                memory_spike_kb=random.randint(1024, 3072),
                cpu_utilization=random.uniform(80, 95),
                custom_fields={
                    "integration_test": True,
                    "scenario": "cpu_spike_anomaly"
                }
            )
            scenario_records.append(record)
        
        # Memory spike anomalies (10%)
        for i in range(5):
            record = TelemetryRecord(
                telemetry_id=f"integration_memory_spike_{i}",
                function_id=f"lambda-function-{i%2}",
                execution_phase="invoke",
                anomaly_type="memory_spike",
                duration=random.uniform(200, 800),
                memory_spike_kb=random.randint(8192, 20480),
                cpu_utilization=random.uniform(30, 60),
                custom_fields={
                    "integration_test": True,
                    "scenario": "memory_spike_anomaly"
                }
            )
            scenario_records.append(record)
        
        # Shuffle to simulate realistic order
        random.shuffle(scenario_records)
        
        print(f"   Processing {len(scenario_records)} telemetry records through full pipeline...")
        
        # Process through full integration pipeline
        integration_results = []
        processing_times = []
        
        for record in scenario_records:
            start_time = time.time()
            
            try:
                # Step 1: Anomaly Detection
                detection_result = anomaly_engine.detect_anomalies(record)
                
                # Step 2: Create telemetry summary for compression
                telemetry_summary = {
                    "telemetry_id": record.telemetry_id,
                    "function_id": record.function_id,
                    "anomaly_score": detection_result.overall_confidence,
                    "detected_anomalies": [
                        {
                            "type": det.anomaly_type,
                            "confidence": det.confidence,
                            "algorithm": det.algorithm_name
                        }
                        for det in detection_result.detections[:3]  # Top 3
                    ],
                    "metrics": {
                        "duration": record.duration,
                        "memory_spike_kb": record.memory_spike_kb,
                        "cpu_utilization": record.cpu_utilization
                    }
                }
                
                # Step 3: Compress telemetry data
                telemetry_json = json.dumps(telemetry_summary).encode()
                compressed_data, compression_metrics = compression_optimizer.compress_data(telemetry_json)
                
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
                
                integration_results.append({
                    "record_id": record.telemetry_id,
                    "original_anomaly_type": record.anomaly_type,
                    "detected_anomaly": detection_result.overall_confidence > 0.5,
                    "anomaly_confidence": detection_result.overall_confidence,
                    "compression_ratio": compression_metrics.compression_ratio,
                    "compression_algorithm": compression_metrics.algorithm.value,
                    "processing_time_ms": processing_time,
                    "pipeline_success": True
                })
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
                
                integration_results.append({
                    "record_id": record.telemetry_id,
                    "original_anomaly_type": record.anomaly_type,
                    "pipeline_success": False,
                    "error": str(e),
                    "processing_time_ms": processing_time
                })
        
        # Analyze integration results
        successful_processing = sum(1 for r in integration_results if r.get("pipeline_success", False))
        success_rate = successful_processing / len(integration_results)
        
        # Anomaly detection accuracy
        detection_results = [r for r in integration_results if r.get("pipeline_success", False)]
        if detection_results:
            true_positives = sum(1 for r in detection_results 
                               if r["original_anomaly_type"] != "benign" and r.get("detected_anomaly", False))
            false_positives = sum(1 for r in detection_results 
                                if r["original_anomaly_type"] == "benign" and r.get("detected_anomaly", False))
            true_negatives = sum(1 for r in detection_results 
                               if r["original_anomaly_type"] == "benign" and not r.get("detected_anomaly", True))
            false_negatives = sum(1 for r in detection_results 
                                if r["original_anomaly_type"] != "benign" and not r.get("detected_anomaly", True))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 0.0
        
        # Performance metrics
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_compression_ratio = sum(r.get("compression_ratio", 1.0) for r in detection_results) / len(detection_results) if detection_results else 1.0
        
        print(f"\n📊 Integration Test Results:")
        print(f"   Pipeline Success Rate: {success_rate*100:.1f}% ({successful_processing}/{len(integration_results)})")
        print(f"   Detection Precision: {precision:.3f}")
        print(f"   Detection Recall: {recall:.3f}")
        print(f"   Detection F1-Score: {f1_score:.3f}")
        print(f"   Average Processing Time: {avg_processing_time:.2f}ms")
        print(f"   Average Compression Ratio: {avg_compression_ratio:.3f}")
        
        # Overall success criteria
        overall_success = (
            success_rate >= 0.9 and  # 90% pipeline success
            f1_score >= 0.7 and      # 70% F1-score for detection
            avg_processing_time < 1000  # Less than 1 second average processing
        )
        
        return {
            "success": overall_success,
            "pipeline_success_rate": success_rate,
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1_score": f1_score,
            "avg_processing_time_ms": avg_processing_time,
            "avg_compression_ratio": avg_compression_ratio,
            "total_records_processed": len(integration_results),
            "successful_records": successful_processing
        }
        
    except Exception as e:
        print(f"❌ Integration scenario test failed: {e}")
        return {"success": False, "error": str(e)}

async def run_comprehensive_system_test():
    """Run comprehensive system test"""
    print("🚀 SCAFAD Layer 0 - Comprehensive Real-World System Test")
    print("=" * 70)
    
    start_time = time.time()
    
    # Phase 1: Component Import Validation
    print("\n" + "="*50)
    print("PHASE 1: COMPONENT IMPORT VALIDATION")
    print("="*50)
    
    import_results = test_system_imports()
    
    if not any(import_results.values()):
        print("❌ CRITICAL: No components could be imported. Cannot proceed with testing.")
        return
    
    # Phase 2: Core Component Testing
    print("\n" + "="*50)
    print("PHASE 2: CORE COMPONENT TESTING")
    print("="*50)
    
    test_results = {}
    
    # Test Anomaly Detection Engine
    if import_results.get('layer0_core', False):
        test_results['anomaly_detection'] = test_anomaly_detection_engine()
    else:
        print("⚠️ Skipping anomaly detection test - component not available")
        test_results['anomaly_detection'] = {"success": False, "reason": "component_unavailable"}
    
    # Test Compression Optimizer
    if import_results.get('layer0_compression_optimizer', False):
        test_results['compression'] = test_compression_optimizer()
    else:
        print("⚠️ Skipping compression test - component not available")
        test_results['compression'] = {"success": False, "reason": "component_unavailable"}
    
    # Test Stream Processor
    if import_results.get('layer0_stream_processor', False):
        test_results['stream_processing'] = await test_stream_processor()
    else:
        print("⚠️ Skipping stream processing test - component not available")
        test_results['stream_processing'] = {"success": False, "reason": "component_unavailable"}
    
    # Test Telemetry Channels
    if import_results.get('app_telemetry', False):
        test_results['telemetry_channels'] = await test_telemetry_channels()
    else:
        print("⚠️ Skipping telemetry channels test - component not available")
        test_results['telemetry_channels'] = {"success": False, "reason": "component_unavailable"}
    
    # Phase 3: Integration Testing
    print("\n" + "="*50)
    print("PHASE 3: FULL SYSTEM INTEGRATION")
    print("="*50)
    
    if import_results.get('layer0_core', False) and import_results.get('layer0_compression_optimizer', False):
        test_results['integration'] = test_integration_scenario()
    else:
        print("⚠️ Skipping integration test - required components not available")
        test_results['integration'] = {"success": False, "reason": "components_unavailable"}
    
    # Phase 4: Results Analysis
    print("\n" + "="*50)
    print("PHASE 4: COMPREHENSIVE RESULTS ANALYSIS")
    print("="*50)
    
    total_time = time.time() - start_time
    
    # Calculate overall metrics
    successful_tests = sum(1 for result in test_results.values() if result.get("success", False))
    total_tests = len(test_results)
    overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 FINAL TEST RESULTS SUMMARY")
    print("-" * 40)
    print(f"Total Test Duration: {total_time:.1f} seconds")
    print(f"Overall Success Rate: {overall_success_rate*100:.1f}% ({successful_tests}/{total_tests})")
    
    # Detailed results per component
    for test_name, result in test_results.items():
        success = result.get("success", False)
        status_icon = "✅" if success else "❌"
        
        if success:
            if test_name == 'anomaly_detection':
                accuracy = result.get('accuracy', 0)
                avg_time = result.get('avg_processing_time_ms', 0)
                print(f"{status_icon} {test_name.replace('_', ' ').title()}: {accuracy:.1f}% accuracy, {avg_time:.2f}ms avg")
            elif test_name == 'compression':
                ratio = result.get('avg_compression_ratio', 1.0)
                throughput = result.get('avg_throughput_mbps', 0)
                print(f"{status_icon} {test_name.replace('_', ' ').title()}: {ratio:.3f} ratio, {throughput:.1f} MB/s")
            elif test_name == 'stream_processing':
                rps = result.get('records_per_second', 0)
                latency = result.get('processing_latency_ms', 0)
                print(f"{status_icon} {test_name.replace('_', ' ').title()}: {rps:.1f} RPS, {latency:.2f}ms latency")
            elif test_name == 'integration':
                f1 = result.get('detection_f1_score', 0)
                pipeline_success = result.get('pipeline_success_rate', 0)
                print(f"{status_icon} {test_name.replace('_', ' ').title()}: {pipeline_success*100:.1f}% pipeline, F1={f1:.3f}")
            else:
                print(f"{status_icon} {test_name.replace('_', ' ').title()}: Success")
        else:
            reason = result.get("reason", result.get("error", "Unknown"))
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: Failed ({reason})")
    
    # System Assessment
    print(f"\n🎯 SYSTEM ASSESSMENT")
    print("-" * 25)
    
    if overall_success_rate >= 0.9:
        print("🟢 EXCELLENT: System performs exceptionally well")
        assessment = "PRODUCTION_READY"
    elif overall_success_rate >= 0.7:
        print("🟡 GOOD: System performs well with minor issues")
        assessment = "PRODUCTION_READY_WITH_MONITORING"
    elif overall_success_rate >= 0.5:
        print("🟠 FAIR: System has some issues but is functional")
        assessment = "NEEDS_IMPROVEMENT"
    else:
        print("🔴 POOR: System has significant issues")
        assessment = "REQUIRES_MAJOR_FIXES"
    
    print(f"Deployment Recommendation: {assessment}")
    
    # Key Insights
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 15)
    
    insights = []
    
    if test_results.get('anomaly_detection', {}).get('success', False):
        accuracy = test_results['anomaly_detection'].get('accuracy', 0)
        if accuracy >= 85:
            insights.append("Anomaly detection accuracy is excellent for production use")
        elif accuracy >= 70:
            insights.append("Anomaly detection accuracy is good but could be improved")
        else:
            insights.append("Anomaly detection accuracy needs significant improvement")
    
    if test_results.get('stream_processing', {}).get('success', False):
        rps = test_results['stream_processing'].get('records_per_second', 0)
        if rps >= 50:
            insights.append("Stream processing throughput is excellent for high-load scenarios")
        elif rps >= 20:
            insights.append("Stream processing throughput is adequate for moderate loads")
        else:
            insights.append("Stream processing throughput may be insufficient for high loads")
    
    if test_results.get('integration', {}).get('success', False):
        f1_score = test_results['integration'].get('detection_f1_score', 0)
        if f1_score >= 0.8:
            insights.append("End-to-end pipeline integration is highly effective")
        elif f1_score >= 0.6:
            insights.append("End-to-end pipeline integration is moderately effective")
        else:
            insights.append("End-to-end pipeline integration needs optimization")
    
    if not insights:
        insights.append("Limited test execution due to component availability issues")
    
    for i, insight in enumerate(insights[:5], 1):  # Show top 5 insights
        print(f"{i}. {insight}")
    
    print(f"\n✨ Test execution completed successfully!")
    print(f"📁 Detailed results available in memory for further analysis")
    
    return {
        "overall_success_rate": overall_success_rate,
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "test_duration_seconds": total_time,
        "assessment": assessment,
        "test_results": test_results,
        "insights": insights
    }

def main():
    """Main test execution function"""
    try:
        # Run the comprehensive test
        result = asyncio.run(run_comprehensive_system_test())
        return result
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return None
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()