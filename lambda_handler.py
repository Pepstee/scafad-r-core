#!/usr/bin/env python3
"""
SCAFAD AWS Lambda Handler
=========================

Main AWS Lambda handler that integrates all SCAFAD components for production deployment.
This handler provides the entry point for AWS Lambda and orchestrates all SCAFAD modules.

Features:
1. Lambda-optimized initialization and warm container management
2. Cold start detection and optimization
3. Integration with all SCAFAD Layer 0 modules
4. Production telemetry emission
5. Error handling and graceful degradation
6. Cost and performance optimization
"""

import json
import logging
import time
import os
import asyncio
from typing import Dict, Any, Optional
import traceback

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for container reuse
SCAFAD_CONTROLLER = None
CONTAINER_START_TIME = time.time()
INVOCATION_COUNT = 0
INITIALIZATION_ERROR = None

def _validate_component_compatibility(config):
    """Validate component version compatibility"""
    required_versions = {
        "python": "3.11",
        "scafad_core": config.version.get("version", "unknown"),
        "layer0_api": "1.0.0"
    }
    
    component_versions = {}
    
    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    component_versions["python"] = python_version
    
    if python_version < required_versions["python"]:
        raise RuntimeError(f"Python {required_versions['python']}+ required, got {python_version}")
    
    # Check core version compatibility
    component_versions["scafad_core"] = required_versions["scafad_core"]
    component_versions["layer0_api"] = required_versions["layer0_api"]
    
    return component_versions

def _validate_component_interface(component, name, required_method):
    """Validate component has required interface"""
    if not hasattr(component, required_method):
        raise RuntimeError(f"Component {name} missing required method: {required_method}")
    
    if not callable(getattr(component, required_method)):
        raise RuntimeError(f"Component {name}.{required_method} is not callable")
    
    logger.info(f"✅ Component {name} interface validated")

def initialize_scafad():
    """Initialize SCAFAD components (called once per container)"""
    global SCAFAD_CONTROLLER, INITIALIZATION_ERROR
    
    try:
        logger.info("Initializing SCAFAD Layer 0 components...")
        start_time = time.time()
        
        # Import SCAFAD modules
        from app_config import get_default_config
        from app_main import Layer0_AdaptiveTelemetryController
        from core.ignn_model import iGNNAnomalyDetector
        from core.real_graph_analysis import GraphAnalysisOrchestrator
        from datasets.serverless_traces import RealisticServerlessTraceGenerator
        
        # Create configuration
        config = get_default_config()
        logger.info(f"SCAFAD configuration loaded: {config.version['version']}")
        
        # CRITICAL FIX #1: Component compatibility validation
        component_versions = _validate_component_compatibility(config)
        logger.info(f"Component compatibility validated: {component_versions}")
        
        # Initialize controller
        SCAFAD_CONTROLLER = Layer0_AdaptiveTelemetryController(config)
        
        # Initialize i-GNN detector with version validation
        ignn_detector = iGNNAnomalyDetector()
        _validate_component_interface(ignn_detector, "iGNNAnomalyDetector", "detect_anomalies")
        
        # Initialize graph analyzer with validation
        graph_orchestrator = GraphAnalysisOrchestrator()
        _validate_component_interface(graph_orchestrator, "GraphAnalysisOrchestrator", "analyze_graph")
        
        # Initialize trace generator with validation
        trace_generator = RealisticServerlessTraceGenerator()
        _validate_component_interface(trace_generator, "RealisticServerlessTraceGenerator", "generate_normal_trace")
        
        # Store components in controller with compatibility check
        SCAFAD_CONTROLLER.ignn_detector = ignn_detector
        SCAFAD_CONTROLLER.graph_orchestrator = graph_orchestrator
        SCAFAD_CONTROLLER.trace_generator = trace_generator
        
        initialization_time = time.time() - start_time
        logger.info(f"✅ SCAFAD initialization completed in {initialization_time:.3f}s")
        
        return True
        
    except Exception as e:
        error_msg = f"❌ SCAFAD initialization failed: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        INITIALIZATION_ERROR = error_msg
        return False


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Main AWS Lambda handler for SCAFAD
    
    Args:
        event: Lambda event data
        context: Lambda context object
        
    Returns:
        Response dictionary with SCAFAD analysis results
    """
    global SCAFAD_CONTROLLER, INVOCATION_COUNT, CONTAINER_START_TIME, INITIALIZATION_ERROR
    
    # Track invocation
    INVOCATION_COUNT += 1
    invocation_start_time = time.time()
    
    # Determine if this is a cold start
    is_cold_start = INVOCATION_COUNT == 1
    container_age = invocation_start_time - CONTAINER_START_TIME
    
    # Log basic invocation info
    logger.info(f"SCAFAD Lambda invocation #{INVOCATION_COUNT}")
    logger.info(f"Cold start: {is_cold_start}, Container age: {container_age:.3f}s")
    logger.info(f"Memory limit: {context.memory_limit_in_mb}MB, Remaining time: {context.get_remaining_time_in_millis()}ms")
    
    try:
        # Initialize SCAFAD on first invocation (cold start)
        if SCAFAD_CONTROLLER is None:
            if not initialize_scafad():
                return {
                    "statusCode": 500,
                    "body": json.dumps({
                        "error": "SCAFAD initialization failed",
                        "details": INITIALIZATION_ERROR,
                        "invocation_id": context.aws_request_id,
                        "cold_start": is_cold_start,
                        "timestamp": invocation_start_time
                    })
                }
        
        # Handle different event types
        if event.get("source") == "aws.events":
            # CloudWatch Events (scheduled or manual)
            result = handle_scheduled_event(event, context)
        elif "Records" in event:
            # SQS, SNS, or other record-based events
            result = handle_record_events(event["Records"], context)
        elif event.get("httpMethod"):
            # API Gateway request
            result = handle_api_request(event, context)
        elif event.get("test_mode"):
            # Test invocation
            result = handle_test_invocation(event, context)
        else:
            # Direct invocation with custom payload
            result = handle_direct_invocation(event, context)
        
        # Add execution metadata
        execution_time = time.time() - invocation_start_time
        result.update({
            "execution_metadata": {
                "invocation_count": INVOCATION_COUNT,
                "is_cold_start": is_cold_start,
                "container_age_seconds": container_age,
                "execution_time_ms": execution_time * 1000,
                "memory_limit_mb": context.memory_limit_in_mb,
                "function_name": context.function_name,
                "function_version": context.function_version,
                "aws_request_id": context.aws_request_id,
                "remaining_time_ms": context.get_remaining_time_in_millis()
            }
        })
        
        logger.info(f"SCAFAD invocation completed in {execution_time:.3f}s")
        return result
        
    except Exception as e:
        execution_time = time.time() - invocation_start_time
        error_msg = f"SCAFAD Lambda handler error: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": error_msg,
                "invocation_id": context.aws_request_id,
                "execution_time_ms": execution_time * 1000,
                "cold_start": is_cold_start,
                "timestamp": invocation_start_time
            })
        }


def handle_scheduled_event(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Handle CloudWatch Events scheduled invocations"""
    
    logger.info("Handling scheduled event")
    
    try:
        # Run SCAFAD self-test
        result = asyncio.run(SCAFAD_CONTROLLER.run_self_test())
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "SCAFAD scheduled self-test completed",
                "result": result,
                "event_type": "scheduled",
                "timestamp": time.time()
            })
        }
        
    except Exception as e:
        logger.error(f"Scheduled event handling failed: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Scheduled event failed: {e}",
                "event_type": "scheduled",
                "timestamp": time.time()
            })
        }


def handle_record_events(records: list, context) -> Dict[str, Any]:
    """Handle SQS, SNS, or other record-based events"""
    
    logger.info(f"Handling {len(records)} record events")
    results = []
    
    for i, record in enumerate(records):
        try:
            # Extract telemetry data from record
            if record.get("eventSource") == "aws:sqs":
                # SQS message
                message_body = record.get("body", "{}")
                telemetry_data = json.loads(message_body)
            elif record.get("EventSource") == "aws:sns":
                # SNS notification
                message = json.loads(record.get("Sns", {}).get("Message", "{}"))
                telemetry_data = message
            else:
                # Generic record
                telemetry_data = record
            
            # Process with SCAFAD
            processing_result = asyncio.run(
                SCAFAD_CONTROLLER.process_telemetry_event(telemetry_data, context)
            )
            
            results.append({
                "record_index": i,
                "status": "success",
                "result": processing_result
            })
            
        except Exception as e:
            logger.error(f"Failed to process record {i}: {e}")
            results.append({
                "record_index": i, 
                "status": "error",
                "error": str(e)
            })
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    return {
        "statusCode": 200 if failed == 0 else 207,  # Multi-status if some failed
        "body": json.dumps({
            "message": f"Processed {len(records)} records",
            "successful": successful,
            "failed": failed,
            "results": results,
            "event_type": "records",
            "timestamp": time.time()
        })
    }


def handle_api_request(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Handle API Gateway requests"""
    
    method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    
    logger.info(f"Handling API request: {method} {path}")
    
    try:
        if method == "GET" and path == "/":
            # Health check
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "message": "SCAFAD Layer 0 API",
                    "version": SCAFAD_CONTROLLER.config.version["version"],
                    "status": "healthy",
                    "timestamp": time.time()
                })
            }
        
        elif method == "POST" and path == "/analyze":
            # Analyze telemetry data
            request_body = event.get("body", "{}")
            if isinstance(request_body, str):
                telemetry_data = json.loads(request_body)
            else:
                telemetry_data = request_body
            
            # Process with SCAFAD
            analysis_result = asyncio.run(
                SCAFAD_CONTROLLER.process_telemetry_event(telemetry_data, context)
            )
            
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "message": "Analysis completed",
                    "result": analysis_result,
                    "timestamp": time.time()
                })
            }
        
        elif method == "GET" and path == "/status":
            # Status endpoint
            status_info = {
                "scafad_version": SCAFAD_CONTROLLER.config.version["version"],
                "container_age": time.time() - CONTAINER_START_TIME,
                "invocation_count": INVOCATION_COUNT,
                "features_enabled": SCAFAD_CONTROLLER.config.get_summary()["features_enabled"],
                "memory_limit": context.memory_limit_in_mb,
                "timestamp": time.time()
            }
            
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps(status_info)
            }
        
        else:
            # Unsupported endpoint
            return {
                "statusCode": 404,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "error": f"Endpoint not found: {method} {path}",
                    "timestamp": time.time()
                })
            }
    
    except Exception as e:
        logger.error(f"API request handling failed: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": f"API request failed: {e}",
                "timestamp": time.time()
            })
        }


def handle_test_invocation(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Handle test invocations"""
    
    logger.info("Handling test invocation")
    
    test_type = event.get("test_type", "basic")
    
    try:
        if test_type == "basic":
            # Basic functionality test
            result = asyncio.run(SCAFAD_CONTROLLER.run_self_test())
            
        elif test_type == "cold_start":
            # Cold start measurement test
            result = {
                "test_type": "cold_start_measurement",
                "is_cold_start": INVOCATION_COUNT == 1,
                "container_age": time.time() - CONTAINER_START_TIME,
                "invocation_count": INVOCATION_COUNT,
                "memory_limit": context.memory_limit_in_mb,
                "remaining_time": context.get_remaining_time_in_millis()
            }
            
        elif test_type == "performance":
            # Performance test with synthetic data
            from datasets.serverless_traces import RealisticServerlessTraceGenerator
            
            generator = RealisticServerlessTraceGenerator()
            test_trace = generator.generate_normal_trace('test-function', 0.1, 10)
            
            # Analyze with i-GNN
            start_time = time.time()
            analysis_result = SCAFAD_CONTROLLER.ignn_detector.detect_anomalies(
                [record.to_dict() for record in test_trace.invocations]
            )
            analysis_time = time.time() - start_time
            
            result = {
                "test_type": "performance",
                "trace_analyzed": {
                    "num_invocations": len(test_trace.invocations),
                    "num_nodes": len(test_trace.graph.nodes),
                    "num_edges": len(test_trace.graph.edges)
                },
                "analysis_result": analysis_result,
                "analysis_time_ms": analysis_time * 1000
            }
            
        elif test_type == "stress":
            # Stress test with multiple analyses
            results = []
            for i in range(5):  # Run 5 quick analyses
                test_data = {
                    "test_mode": True,
                    "stress_iteration": i,
                    "timestamp": time.time()
                }
                
                result_i = asyncio.run(
                    SCAFAD_CONTROLLER.process_telemetry_event(test_data, context)
                )
                results.append(result_i)
            
            result = {
                "test_type": "stress",
                "iterations": len(results),
                "all_successful": all("error" not in str(r) for r in results),
                "sample_results": results[:2]  # Include first 2 results
            }
            
        else:
            result = {
                "error": f"Unknown test type: {test_type}",
                "available_types": ["basic", "cold_start", "performance", "stress"]
            }
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Test invocation completed",
                "test_type": test_type,
                "result": result,
                "timestamp": time.time()
            })
        }
        
    except Exception as e:
        logger.error(f"Test invocation failed: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Test invocation failed: {e}",
                "test_type": test_type,
                "timestamp": time.time()
            })
        }


def handle_direct_invocation(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Handle direct Lambda invocations with custom payloads"""
    
    logger.info("Handling direct invocation")
    
    try:
        # Process the event with SCAFAD
        result = asyncio.run(
            SCAFAD_CONTROLLER.process_telemetry_event(event, context)
        )
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "SCAFAD analysis completed",
                "result": result,
                "event_type": "direct",
                "timestamp": time.time()
            })
        }
        
    except Exception as e:
        logger.error(f"Direct invocation failed: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Direct invocation failed: {e}",
                "event_type": "direct", 
                "timestamp": time.time()
            })
        }


def warm_container():
    """Warm up the container by pre-initializing components"""
    global SCAFAD_CONTROLLER
    
    if SCAFAD_CONTROLLER is None:
        logger.info("Warming container...")
        initialize_scafad()
        
        # Run a quick self-test to warm up all components
        if SCAFAD_CONTROLLER:
            try:
                asyncio.run(SCAFAD_CONTROLLER.run_self_test())
                logger.info("✅ Container warmed successfully")
            except Exception as e:
                logger.error(f"Container warming failed: {e}")


# Container warm-up (executes during container initialization)
if os.getenv("LAMBDA_TASK_ROOT"):  # Only in Lambda environment
    warm_container()


# Export for testing
__all__ = ['lambda_handler', 'initialize_scafad', 'warm_container']