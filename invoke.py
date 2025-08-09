#!/usr/bin/env python3
"""
SCAFAD Layer 0 Enhanced Invocation Script - SAM CLI Version
Complete rewrite for proper SAM CLI integration with status code 202 support
Version: v4.2-sam-complete
"""

import json
import random
import subprocess
import time
import string
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any
import uuid
import hashlib

# Enhanced argument parsing
parser = argparse.ArgumentParser(
    description="Invoke SCAFAD Layer 0 Lambda with comprehensive test payloads via SAM CLI",
    epilog="Example: python invoke.py --n 10 --seed 42 --mode test --adversarial"
)
parser.add_argument("--n", type=int, default=10, help="Number of invocations to simulate (default: 10)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
parser.add_argument("--mode", choices=['test', 'production', 'demo'], default='test', 
                   help="Invocation mode (default: test)")
parser.add_argument("--adversarial", action='store_true', 
                   help="Enable adversarial attack simulation")
parser.add_argument("--economic", action='store_true', 
                   help="Include economic abuse scenarios")
parser.add_argument("--verbose", "-v", action='store_true', 
                   help="Verbose output")
parser.add_argument("--output-dir", default="telemetry", 
                   help="Output directory for payloads (default: telemetry)")
parser.add_argument("--function-name", default="HelloWorldFunction", 
                   help="Lambda function name (default: HelloWorldFunction)")
parser.add_argument("--delay", type=float, default=0.5, 
                   help="Delay between invocations in seconds (default: 0.5)")
parser.add_argument("--batch-size", type=int, default=0, 
                   help="Process in batches (0 = no batching)")
parser.add_argument("--sam-timeout", type=int, default=120, 
                   help="SAM CLI timeout in seconds (default: 120)")

args = parser.parse_args()

# Configuration
N = args.n
SEED = args.seed
MODE = args.mode
ENABLE_ADVERSARIAL = args.adversarial
ENABLE_ECONOMIC = args.economic
VERBOSE = args.verbose
OUTPUT_DIR = args.output_dir
FUNCTION_NAME = args.function_name
DELAY = args.delay
BATCH_SIZE = max(1, args.batch_size) if args.batch_size > 0 else N
SAM_TIMEOUT = args.sam_timeout

# Set random seed for reproducibility
random.seed(SEED)

# Complete SCAFAD Layer 0 Anomaly Types (All supported)
ANOMALY_TYPES = [
    "benign", "cold_start", "cpu_burst", "memory_spike", 
    "io_intensive", "network_anomaly", "starvation_fallback", 
    "timeout_fallback", "execution_failure", "adversarial_injection"
]

# Execution phases
EXECUTION_PHASES = ["init", "invoke", "shutdown"]

# Function profiles for comprehensive testing
FUNCTION_PROFILES = [
    "ml_inference", "data_processor", "api_gateway", 
    "auth_service", "file_processor", "notification_service",
    "analytics_engine", "cache_manager", "image_resizer",
    "log_aggregator", "stream_processor", "batch_worker"
]

# Complete adversarial attack vectors
ADVERSARIAL_ATTACKS = [
    "adaptive", "dos_amplification", "billing_attack", 
    "cryptomining", "resource_exhaustion", "cold_start_exploitation",
    "memory_bomb", "cpu_exhaustion", "network_flooding",
    "timing_attack", "side_channel", "privilege_escalation"
]

# Economic abuse scenarios
ECONOMIC_ATTACKS = [
    "billing_amplification", "resource_waste", "concurrent_abuse",
    "memory_bomb", "timeout_exploitation", "init_spam",
    "cold_start_farming", "duration_maximization", "invocation_flooding"
]

# Starvation simulation patterns
STARVATION_PATTERNS = [
    "resource_contention", "memory_pressure", "cpu_starvation",
    "io_bottleneck", "network_congestion", "dependency_failure"
]

def check_sam_prerequisites():
    """Check if SAM CLI and Docker are available and working"""
    print("🔍 Checking SAM CLI prerequisites...")
    
    # Check SAM CLI
    try:
        result = subprocess.run(['sam', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ SAM CLI: {result.stdout.strip()}")
        else:
            print("❌ SAM CLI not working properly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ SAM CLI not found or not responding")
        return False
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker not working properly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Docker not found or not responding")
        return False
    
    # Check if Docker daemon is running
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
        else:
            print("❌ Docker daemon not running")
            print("💡 Start Docker Desktop and try again")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker daemon check timed out")
        return False
    
    return True

def create_output_directories():
    """Create comprehensive output directory structure"""
    directories = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/payloads",
        f"{OUTPUT_DIR}/responses",
        f"{OUTPUT_DIR}/logs",
        f"{OUTPUT_DIR}/analysis",
        f"{OUTPUT_DIR}/graphs",
        f"{OUTPUT_DIR}/adversarial",
        f"{OUTPUT_DIR}/economic"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"📁 Created output directories in: {OUTPUT_DIR}")

def is_successful_lambda_response(json_response):
    """
    Check if Lambda response indicates success
    Accepts both 200 (OK) and 202 (Accepted) as success for SCAFAD
    """
    if not json_response or 'statusCode' not in json_response:
        return False
    
    # SCAFAD returns 202 for asynchronous telemetry processing
    # This is the correct status code for background processing
    return json_response['statusCode'] in [200, 202]

def generate_enhanced_payload(index: int) -> Dict:
    """Generate comprehensive enhanced payload for SCAFAD Layer 0"""
    
    # Basic payload structure
    anomaly = random.choice(ANOMALY_TYPES)
    profile = random.choice(FUNCTION_PROFILES)
    phase = random.choice(EXECUTION_PHASES)
    concurrency_id = ''.join(random.choices(string.ascii_uppercase, k=3))
    
    # Base SCAFAD payload
    payload = {
        # Core SCAFAD Layer 0 fields
        "anomaly": anomaly,
        "function_profile_id": profile,
        "execution_phase": phase,
        "concurrency_id": concurrency_id,
        "invocation_timestamp": time.time(),
        "test_mode": MODE == 'test',
        
        # Payload identification
        "payload_id": f"scafad_l0_{index:04d}",
        "batch_id": f"batch_{int(time.time())}_{random.randint(1000, 9999)}",
        "seed": SEED,
        "invocation_index": index,
        
        # Layer 0 Enhanced Features
        "layer0_enhanced": True,
        "schema_version": "v4.2",
        "enable_graph_analysis": random.choice([True, False]),
        "enable_provenance": True,
        "enable_economic_monitoring": ENABLE_ECONOMIC,
        "enable_adversarial_detection": ENABLE_ADVERSARIAL,
        
        # Execution environment simulation
        "execution_environment": {
            "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
            "runtime": "python3.11",
            "memory_allocation": random.choice([128, 256, 512, 1024]),
            "architecture": "x86_64"
        },
        
        # Performance tracking
        "performance_targets": {
            "max_duration": random.uniform(1.0, 30.0),
            "max_memory_mb": random.randint(64, 1024),
            "expected_latency": random.uniform(0.1, 5.0)
        },
        
        # Starvation simulation (10% chance)
        "force_starvation": random.choice([True] + [False] * 9),
        
        # Metadata
        "metadata": {
            "generator": "scafad_sam_invoke.py",
            "generation_time": datetime.now().isoformat(),
            "mode": MODE,
            "total_invocations": N,
            "invocation_index": index
        }
    }
    
    # Add adversarial configuration if enabled
    if ENABLE_ADVERSARIAL and random.random() < 0.35:  # 35% chance
        attack_type = random.choice(ADVERSARIAL_ATTACKS)
        payload.update({
            "enable_adversarial": True,
            "attack_type": attack_type,
            "adversarial_intensity": random.uniform(0.1, 1.0),
            "adversarial_metadata": {
                "target": profile,
                "expected_impact": random.choice(["low", "medium", "high"]),
                "evasion_strategy": random.choice(["gradual", "burst", "stealth"])
            }
        })
    
    # Add economic attack patterns if enabled
    if ENABLE_ECONOMIC and random.random() < 0.25:  # 25% chance
        economic_attack = random.choice(ECONOMIC_ATTACKS)
        payload.update({
            "economic_attack": economic_attack,
            "cost_impact_target": random.uniform(1.5, 10.0),
            "economic_metadata": {
                "attack_duration": random.randint(60, 1800),
                "resource_amplification": random.uniform(2.0, 5.0)
            }
        })
    
    return payload

def save_payload(payload: Dict, index: int) -> str:
    """Save payload to file with enhanced categorization"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    anomaly = payload.get("anomaly", "unknown")
    
    # Categorize payload for better organization
    category = "adversarial" if payload.get("enable_adversarial") else \
               "economic" if payload.get("economic_attack") else \
               "normal"
    
    filename = f"{OUTPUT_DIR}/payloads/payload_{index:04d}_{anomaly}_{category}_{timestamp}.json"
    
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    return filename

def invoke_lambda_with_sam(payload: Dict, index: int) -> Dict:
    """
    Invoke the Lambda function using SAM CLI with enhanced error handling
    and proper status code 202 recognition
    """
    
    # Save payload to temporary file for SAM
    temp_payload_file = f"temp_payload_{index}.json"
    with open(temp_payload_file, "w", encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    start_time = time.time()
    
    try:
        # Build SAM command
        cmd = ["sam", "local", "invoke", FUNCTION_NAME, "--event", temp_payload_file]
        
        if VERBOSE:
            print(f"   🚀 Executing: {' '.join(cmd)}")
        
        # Execute SAM command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SAM_TIMEOUT
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Initialize response object
        response = {
            "invocation_index": index,
            "success": False,  # Will be updated based on Lambda response
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat(),
            "payload_id": payload.get("payload_id", f"sam_{index}")
        }
        
        # Parse Lambda response from stdout
        json_response = None
        if result.returncode == 0:
            # Look for JSON response in stdout
            stdout_lines = result.stdout.split('\n')
            for line in stdout_lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        json_response = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if json_response:
                response["lambda_response"] = json_response
                
                # ENHANCED SUCCESS CHECK - Accept both 200 and 202
                response["success"] = is_successful_lambda_response(json_response)
                
                # Extract SCAFAD-specific metrics if available
                if isinstance(json_response.get("body"), str):
                    try:
                        body = json.loads(json_response["body"])
                        response["scafad_metrics"] = {
                            "anomaly_detected": body.get("anomaly_detected"),
                            "telemetry_id": body.get("telemetry_id"),
                            "processing_time_ms": body.get("processing_time_ms"),
                            "economic_risk_score": body.get("economic_risk_score"),
                            "completeness_score": body.get("completeness_score")
                        }
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(json_response.get("body"), dict):
                    # Direct body object
                    body = json_response["body"]
                    response["scafad_metrics"] = {
                        "anomaly_detected": body.get("anomaly_detected"),
                        "telemetry_id": body.get("telemetry_id"),
                        "processing_time_ms": body.get("processing_time_ms"),
                        "economic_risk_score": body.get("economic_risk_score"),
                        "completeness_score": body.get("completeness_score")
                    }
            else:
                # No JSON response found, but subprocess succeeded
                if VERBOSE:
                    print(f"   ⚠️  No JSON response found in stdout")
                response["success"] = False
        else:
            # SAM command failed
            response["success"] = False
            if VERBOSE:
                print(f"   ❌ SAM command failed with return code: {result.returncode}")
                print(f"   📝 STDERR: {result.stderr[:200]}...")
        
        # Clean up temporary file
        try:
            os.remove(temp_payload_file)
        except OSError:
            pass
        
        return response
        
    except subprocess.TimeoutExpired:
        # Clean up temporary file
        try:
            os.remove(temp_payload_file)
        except OSError:
            pass
        
        return {
            "invocation_index": index,
            "success": False,
            "duration": SAM_TIMEOUT,
            "error": "SAM CLI timeout",
            "error_type": "timeout",
            "timestamp": datetime.now().isoformat(),
            "payload_id": payload.get("payload_id", f"timeout_{index}")
        }
    
    except Exception as e:
        # Clean up temporary file
        try:
            os.remove(temp_payload_file)
        except OSError:
            pass
        
        return {
            "invocation_index": index,
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
            "payload_id": payload.get("payload_id", f"error_{index}")
        }

def save_response(response: Dict, index: int):
    """Save invocation response with categorization"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    success = "success" if response.get("success") else "failure"
    
    filename = f"{OUTPUT_DIR}/responses/response_{index:04d}_{success}_{timestamp}.json"
    
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(response, f, indent=2, ensure_ascii=False)

def update_master_log(payload: Dict, response: Dict):
    """Update comprehensive master log"""
    master_log_path = f"{OUTPUT_DIR}/invocation_master_log.jsonl"
    
    # Extract SCAFAD metrics from response
    scafad_metrics = response.get("scafad_metrics", {})
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "payload_summary": {
            "payload_id": payload.get("payload_id"),
            "anomaly": payload.get("anomaly"),
            "function_profile": payload.get("function_profile_id"),
            "execution_phase": payload.get("execution_phase"),
            "adversarial": payload.get("enable_adversarial", False),
            "economic_attack": payload.get("economic_attack"),
            "graph_analysis": payload.get("enable_graph_analysis", False),
            "starvation_forced": payload.get("force_starvation", False)
        },
        "response_summary": {
            "success": response.get("success"),
            "duration": response.get("duration"),
            "status_code": response.get("lambda_response", {}).get("statusCode"),
            "error_type": response.get("error_type"),
            "timestamp": response.get("timestamp")
        },
        "scafad_analysis": {
            "anomaly_detected": scafad_metrics.get("anomaly_detected"),
            "processing_time_ms": scafad_metrics.get("processing_time_ms"),
            "economic_risk_score": scafad_metrics.get("economic_risk_score"),
            "completeness_score": scafad_metrics.get("completeness_score"),
            "telemetry_id": scafad_metrics.get("telemetry_id")
        },
        "full_payload": payload,
        "full_response": response
    }
    
    with open(master_log_path, "a", encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def print_invocation_summary(payload: Dict, response: Dict, index: int):
    """Print enhanced invocation summary with SCAFAD metrics"""
    anomaly = payload.get("anomaly", "unknown")
    profile = payload.get("function_profile_id", "unknown")
    phase = payload.get("execution_phase", "unknown")
    success = response.get("success", False)
    duration = response.get("duration", 0)
    
    # Status and feature icons
    status_icon = "✅" if success else "❌"
    adversarial_icon = "🎭" if payload.get("enable_adversarial") else ""
    economic_icon = "💰" if payload.get("economic_attack") else ""
    graph_icon = "🕸️" if payload.get("enable_graph_analysis") else ""
    starvation_icon = "⚡" if payload.get("force_starvation") else ""
    
    print(f"{status_icon} [{index+1:3d}/{N}] {profile} | {anomaly} | {phase} | {duration:.2f}s {adversarial_icon}{economic_icon}{graph_icon}{starvation_icon}")
    
    if VERBOSE and response.get("lambda_response"):
        lambda_resp = response["lambda_response"]
        status_code = lambda_resp.get("statusCode", "unknown")
        print(f"         └─ Lambda Status: {status_code}")
        
        # Enhanced SCAFAD metrics display
        scafad_metrics = response.get("scafad_metrics", {})
        if scafad_metrics:
            anomaly_detected = scafad_metrics.get("anomaly_detected")
            if anomaly_detected is not None:
                detection_icon = "🚨" if anomaly_detected else "🟢"
                print(f"         └─ Anomaly Detected: {detection_icon} {anomaly_detected}")
            
            processing_time = scafad_metrics.get("processing_time_ms")
            if processing_time:
                print(f"         └─ Processing Time: ⏱️ {processing_time:.2f}ms")
            
            risk_score = scafad_metrics.get("economic_risk_score")
            if risk_score is not None:
                risk_icon = "🔴" if risk_score > 0.7 else "🟡" if risk_score > 0.3 else "🟢"
                print(f"         └─ Economic Risk: {risk_icon} {risk_score:.2f}")
    
    elif not success and VERBOSE:
        error = response.get("error", "Unknown error")
        print(f"         └─ Error: {error}")

def print_execution_summary(start_time: float, responses: List[Dict]):
    """Print comprehensive execution summary with SCAFAD analytics"""
    end_time = time.time()
    total_duration = end_time - start_time
    
    successful = sum(1 for r in responses if r.get("success", False))
    failed = len(responses) - successful
    
    if responses:
        avg_duration = sum(r.get("duration", 0) for r in responses) / len(responses)
        min_duration = min(r.get("duration", 0) for r in responses)
        max_duration = max(r.get("duration", 0) for r in responses)
    else:
        avg_duration = min_duration = max_duration = 0
    
    print(f"\n📊 Execution Summary")
    print(f"{'='*60}")
    print(f"🎯 Total Invocations: {N}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {successful/N*100:.1f}%")
    print(f"⏱️  Total Time: {total_duration:.2f}s")
    print(f"⚡ Avg Response Time: {avg_duration:.2f}s")
    print(f"🏃 Min Response Time: {min_duration:.2f}s")
    print(f"🐌 Max Response Time: {max_duration:.2f}s")
    
    # SCAFAD-specific analytics
    anomaly_detections = 0
    total_processing_time = 0
    high_risk_count = 0
    
    for response in responses:
        scafad_metrics = response.get("scafad_metrics", {})
        if scafad_metrics.get("anomaly_detected"):
            anomaly_detections += 1
        if scafad_metrics.get("processing_time_ms"):
            total_processing_time += scafad_metrics["processing_time_ms"]
        if scafad_metrics.get("economic_risk_score", 0) > 0.7:
            high_risk_count += 1
    
    if successful > 0:
        print(f"\n🔍 SCAFAD Layer 0 Analytics:")
        print(f"🚨 Anomalies Detected: {anomaly_detections}/{successful} ({anomaly_detections/successful*100:.1f}%)")
        if total_processing_time > 0:
            print(f"⚙️  Avg Processing Time: {total_processing_time/successful:.2f}ms")
        print(f"🔴 High Risk Invocations: {high_risk_count}/{successful} ({high_risk_count/successful*100:.1f}%)")
    
    print(f"\n📁 Output Files:")
    print(f"├── 📋 Payloads: {OUTPUT_DIR}/payloads/")
    print(f"├── 📤 Responses: {OUTPUT_DIR}/responses/")
    print(f"├── 📃 Master Log: {OUTPUT_DIR}/invocation_master_log.jsonl")
    print(f"├── 📊 Analysis: {OUTPUT_DIR}/analysis/")
    print(f"├── 🕸️  Graphs: {OUTPUT_DIR}/graphs/")
    print(f"├── 🎭 Adversarial: {OUTPUT_DIR}/adversarial/")
    print(f"└── 💰 Economic: {OUTPUT_DIR}/economic/")

def main():
    """Main execution function with comprehensive SCAFAD Layer 0 testing via SAM CLI"""
    print(f"🚀 SCAFAD Layer 0 Enhanced Invocation Script - SAM CLI Version")
    print(f"{'='*70}")
    
    # Check prerequisites first
    if not check_sam_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        print("💡 Make sure Docker Desktop is running and SAM CLI is installed")
        return False
    
    print(f"\n📊 Configuration:")
    print(f"   • Invocations: {N}")
    print(f"   • Mode: {MODE}")
    print(f"   • Adversarial: {'✅' if ENABLE_ADVERSARIAL else '❌'}")
    print(f"   • Economic Attacks: {'✅' if ENABLE_ECONOMIC else '❌'}")
    print(f"   • Function: {FUNCTION_NAME}")
    print(f"   • Delay: {DELAY}s")
    print(f"   • SAM Timeout: {SAM_TIMEOUT}s")
    print(f"   • Output: {OUTPUT_DIR}/")
    print(f"   • Seed: {SEED}")
    print(f"   • Batch Size: {BATCH_SIZE}")
    
    # Create comprehensive output directories
    create_output_directories()
    
    # Generate all enhanced payloads
    print(f"\n📦 Generating {N} enhanced SCAFAD payloads...")
    payloads = []
    
    for i in range(N):
        payload = generate_enhanced_payload(i)
        payloads.append(payload)
    
    print(f"✅ Generated {len(payloads)} payloads")
    
    # Payload composition analysis
    anomaly_counts = {}
    for payload in payloads:
        anomaly = payload.get("anomaly", "unknown")
        anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
    
    print(f"   📊 Anomaly Distribution:")
    for anomaly, count in sorted(anomaly_counts.items()):
        percentage = count / len(payloads) * 100
        print(f"      • {anomaly}: {count} ({percentage:.1f}%)")
    
    # Execute invocations
    start_time = time.time()
    print(f"\n🚀 Starting SCAFAD Layer 0 invocations...")
    
    all_responses = []
    
    # Process each payload
    for i, payload in enumerate(payloads):
        # Save payload
        save_payload(payload, i)
        
        # Invoke function via SAM CLI
        response = invoke_lambda_with_sam(payload, i)
        
        # Save response
        save_response(response, i)
        
        # Update master log
        update_master_log(payload, response)
        
        # Print summary
        print_invocation_summary(payload, response, i)
        
        all_responses.append(response)
        
        # Delay between invocations
        if i < len(payloads) - 1:
            time.sleep(DELAY)
    
    # Print comprehensive final summary
    print_execution_summary(start_time, all_responses)
    
    # Generate final report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{OUTPUT_DIR}/analysis/execution_report_{timestamp}.json"
    
    report_data = {
        "execution_summary": {
            "total_invocations": N,
            "successful": sum(1 for r in all_responses if r.get("success")),
            "failed": sum(1 for r in all_responses if not r.get("success")),
            "total_duration": time.time() - start_time,
            "success_rate": sum(1 for r in all_responses if r.get("success")) / N if N > 0 else 0
        },
        "configuration": {
            "mode": MODE,
            "adversarial_enabled": ENABLE_ADVERSARIAL,
            "economic_enabled": ENABLE_ECONOMIC,
            "seed": SEED,
            "function_name": FUNCTION_NAME,
            "sam_timeout": SAM_TIMEOUT
        },
        "detailed_results": all_responses
    }
    
    with open(report_filename, "w", encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Detailed report saved: {report_filename}")
    print(f"\n🎉 All {N} SCAFAD Layer 0 invocations completed!")
    print(f"💡 Next steps:")
    print(f"   1. Check logs: python fetch_logs.py")
    print(f"   2. Analyze results: {OUTPUT_DIR}/analysis/")
    print(f"   3. Review master log: {OUTPUT_DIR}/invocation_master_log.jsonl")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  SCAFAD invocation interrupted by user")
        print(f"📁 Partial results may be available in: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"\n❌ SCAFAD invocation failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check SAM CLI installation: sam --version")
        print(f"   2. Verify Docker is running: docker ps")
        print(f"   3. Build your function: sam build")
        print(f"   4. Test manual invoke: sam local invoke {FUNCTION_NAME} --event event.json")
        print(f"   5. Check function template.yaml")
        print(f"   6. Ensure all dependencies are installed: pip install tenacity")
        exit(1)