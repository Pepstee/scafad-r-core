# invoke.py
"""
Enhanced invocation script for SCAFAD Layer 0
Generates comprehensive test payloads and invokes the Lambda function
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

# Enhanced argument parsing
parser = argparse.ArgumentParser(
    description="Invoke SCAFAD Layer 0 Lambda with comprehensive test payloads",
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
BATCH_SIZE = args.batch_size

# Set random seed for reproducibility
random.seed(SEED)

# Enhanced test configurations
ANOMALY_TYPES = [
    "benign", "cold_start", "cpu_burst", "memory_spike", 
    "io_intensive", "network_anomaly", "execution_failure"
]

FUNCTION_PROFILES = [
    "data_processor", "ml_inference", "api_gateway", "file_processor",
    "notification_service", "auth_service", "analytics_engine",
    "image_resizer", "log_aggregator", "cache_manager"
]

EXECUTION_PHASES = ["init", "invoke", "shutdown"]

# Economic attack patterns (if enabled)
ECONOMIC_ATTACKS = [
    "dos_amplification", "billing_attack", "cryptomining", 
    "data_exfiltration", "privilege_escalation", "resource_exhaustion"
]

# Adversarial configurations
ADVERSARIAL_CONFIGS = [
    {"attack_type": "adaptive", "intensity": 0.5},
    {"attack_type": "dos_amplification", "intensity": 0.7},
    {"attack_type": "billing_attack", "intensity": 0.6},
    {"attack_type": "cryptomining", "intensity": 0.8},
    {"attack_type": "resource_exhaustion", "intensity": 0.9}
]

def create_output_directories():
    """Create necessary output directories"""
    directories = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/payloads",
        f"{OUTPUT_DIR}/responses",
        f"{OUTPUT_DIR}/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    if VERBOSE:
        print(f"📁 Created output directories in: {OUTPUT_DIR}")

def generate_realistic_payload_data(profile: str, anomaly: str) -> Dict:
    """Generate realistic payload data based on function profile and anomaly"""
    
    base_payloads = {
        "data_processor": {
            "Records": [
                {
                    "eventSource": "aws:s3",
                    "s3": {
                        "bucket": {"name": "data-lake-bucket"},
                        "object": {"key": f"data/input_{random.randint(1000, 9999)}.json"}
                    }
                }
            ],
            "processing_config": {
                "batch_size": random.randint(100, 1000),
                "format": random.choice(["json", "csv", "parquet"])
            }
        },
        
        "ml_inference": {
            "model_name": f"model_{random.choice(['sentiment', 'classification', 'regression'])}",
            "input_data": {
                "features": [random.uniform(0, 1) for _ in range(random.randint(10, 50))],
                "metadata": {"version": "v1.2", "timestamp": time.time()}
            },
            "inference_config": {
                "batch_inference": random.choice([True, False]),
                "confidence_threshold": random.uniform(0.7, 0.95)
            }
        },
        
        "api_gateway": {
            "httpMethod": random.choice(["GET", "POST", "PUT", "DELETE"]),
            "path": f"/api/v1/{random.choice(['users', 'orders', 'products', 'analytics'])}",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {uuid.uuid4()}",
                "User-Agent": "SCAFAD-Test-Client/1.0"
            },
            "queryStringParameters": {
                "limit": str(random.randint(10, 100)),
                "offset": str(random.randint(0, 1000))
            },
            "body": json.dumps({
                "action": random.choice(["create", "update", "delete", "query"]),
                "data": {"id": random.randint(1, 10000)}
            })
        },
        
        "file_processor": {
            "file_info": {
                "name": f"document_{random.randint(1000, 9999)}.pdf",
                "size": random.randint(1024, 10485760),  # 1KB to 10MB
                "type": random.choice(["pdf", "docx", "txt", "csv"])
            },
            "processing_options": {
                "extract_text": True,
                "generate_thumbnail": random.choice([True, False]),
                "compress": random.choice([True, False])
            }
        },
        
        "notification_service": {
            "notification_type": random.choice(["email", "sms", "push", "webhook"]),
            "recipients": [f"user{i}@example.com" for i in range(random.randint(1, 5))],
            "message": {
                "subject": "Test Notification",
                "body": "This is a test notification from SCAFAD Layer 0",
                "priority": random.choice(["low", "normal", "high", "urgent"])
            },
            "delivery_options": {
                "retry_attempts": random.randint(1, 5),
                "delay_seconds": random.randint(0, 300)
            }
        }
    }
    
    # Get base payload or create a generic one
    payload = base_payloads.get(profile, {
        "generic_data": {"type": profile, "data": f"test_data_{random.randint(1, 1000)}"},
        "metadata": {"generated": True}
    })
    
    # Add anomaly-specific modifications
    if anomaly == "memory_spike":
        payload["large_data"] = "x" * random.randint(1000, 10000)
    elif anomaly == "cpu_burst":
        payload["computational_task"] = {
            "iterations": random.randint(10000, 100000),
            "complexity": "high"
        }
    elif anomaly == "io_intensive":
        payload["io_operations"] = [
            {"type": "read", "size": random.randint(1024, 102400)} 
            for _ in range(random.randint(5, 20))
        ]
    elif anomaly == "network_anomaly":
        payload["network_calls"] = [
            {"url": f"https://api{i}.example.com", "timeout": random.randint(1, 30)}
            for i in range(random.randint(3, 10))
        ]
    
    return payload

def generate_enhanced_payload(index: int) -> Dict:
    """Generate enhanced payload for SCAFAD Layer 0"""
    
    # Basic payload structure
    anomaly = random.choice(ANOMALY_TYPES)
    profile = random.choice(FUNCTION_PROFILES)
    phase = random.choice(EXECUTION_PHASES)
    concurrency_id = ''.join(random.choices(string.ascii_uppercase, k=3))
    
    # Generate realistic payload data
    payload_data = generate_realistic_payload_data(profile, anomaly)
    
    # Base payload
    payload = {
        # Core SCAFAD fields
        "anomaly": anomaly,
        "function_profile_id": profile,
        "execution_phase": phase,
        "concurrency_id": concurrency_id,
        "invocation_timestamp": time.time(),
        "test_mode": MODE == 'test',
        
        # Payload identification
        "payload_id": f"scafad_invoke_{index:04d}",
        "batch_id": f"batch_{int(time.time())}",
        "seed": SEED,
        
        # Realistic payload data
        **payload_data,
        
        # Layer 0 specific flags
        "layer0_enhanced": True,
        "schema_version": "v4.2",
        "enable_graph_analysis": True,
        "enable_provenance": True,
        "enable_economic_monitoring": ENABLE_ECONOMIC,
        
        # Starvation simulation (occasional)
        "force_starvation": random.choice([True] + [False] * 9),  # 10% chance
        
        # Metadata
        "metadata": {
            "generator": "enhanced_invoke.py",
            "generation_time": datetime.now().isoformat(),
            "mode": MODE,
            "total_invocations": N,
            "invocation_index": index
        }
    }
    
    # Add adversarial configuration if enabled
    if ENABLE_ADVERSARIAL and random.random() < 0.3:  # 30% chance
        adv_config = random.choice(ADVERSARIAL_CONFIGS)
        payload.update({
            "enable_adversarial": True,
            "attack_type": adv_config["attack_type"],
            "adversarial_intensity": adv_config["intensity"],
            "adversarial_metadata": {
                "target": profile,
                "expected_impact": random.choice(["low", "medium", "high"]),
                "evasion_strategy": random.choice(["gradual", "burst", "stealth"])
            }
        })
    
    # Add economic attack patterns if enabled
    if ENABLE_ECONOMIC and random.random() < 0.2:  # 20% chance
        economic_attack = random.choice(ECONOMIC_ATTACKS)
        payload.update({
            "economic_attack": economic_attack,
            "cost_impact_target": random.uniform(1.5, 10.0),  # 1.5x to 10x cost
            "billing_abuse_type": random.choice(["duration", "memory", "invocation_count"])
        })
    
    # Add parent chain for some invocations (function chaining)
    if random.random() < 0.15:  # 15% chance
        payload["parent_chain"] = f"parent_chain_{random.randint(1000, 9999)}"
    
    # Add simulation of different execution environments
    payload["execution_environment"] = {
        "region": random.choice(["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]),
        "availability_zone": random.choice(["a", "b", "c"]),
        "runtime": random.choice(["python3.11", "python3.10", "python3.9"]),
        "memory_allocation": random.choice([128, 256, 512, 1024, 2048])
    }
    
    return payload

def save_payload(payload: Dict, index: int) -> str:
    """Save payload to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{OUTPUT_DIR}/payloads/payload_{index:04d}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    
    return filename

def invoke_lambda_function(payload: Dict, index: int) -> Dict:
    """Invoke the Lambda function with the payload"""
    
    # Save payload to shared file for SAM
    with open("payload.json", "w") as f:
        json.dump(payload, f, indent=2)
    
    start_time = time.time()
    
    try:
        # Invoke using SAM CLI
        cmd = ["sam", "local", "invoke", FUNCTION_NAME, "--event", "payload.json"]
        
        if VERBOSE:
            print(f"   🚀 Executing: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse response
        response = {
            "invocation_index": index,
            "success": result.returncode == 0,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to extract JSON response from stdout
        try:
            # SAM output includes extra text, find the JSON part
            stdout_lines = result.stdout.split('\n')
            for line in stdout_lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    response["lambda_response"] = json.loads(line.strip())
                    break
        except json.JSONDecodeError:
            if VERBOSE:
                print(f"   ⚠️  Could not parse Lambda response JSON")
        
        return response
        
    except subprocess.TimeoutExpired:
        return {
            "invocation_index": index,
            "success": False,
            "duration": 120,
            "error": "Timeout",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "invocation_index": index,
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def save_response(response: Dict, index: int):
    """Save invocation response"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{OUTPUT_DIR}/responses/response_{index:04d}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(response, f, indent=2)

def update_master_log(payload: Dict, response: Dict):
    """Update master log with invocation details"""
    master_log_path = f"{OUTPUT_DIR}/invocation_master_log.jsonl"
    
    log_entry = {
        "payload_summary": {
            "anomaly": payload.get("anomaly"),
            "function_profile": payload.get("function_profile_id"),
            "execution_phase": payload.get("execution_phase"),
            "adversarial": payload.get("enable_adversarial", False),
            "economic_attack": payload.get("economic_attack"),
            "payload_id": payload.get("payload_id")
        },
        "response_summary": {
            "success": response.get("success"),
            "duration": response.get("duration"),
            "status_code": response.get("lambda_response", {}).get("statusCode"),
            "timestamp": response.get("timestamp")
        },
        "full_payload": payload,
        "full_response": response
    }
    
    with open(master_log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def print_invocation_summary(payload: Dict, response: Dict, index: int):
    """Print summary of invocation"""
    anomaly = payload.get("anomaly", "unknown")
    profile = payload.get("function_profile_id", "unknown")
    phase = payload.get("execution_phase", "unknown")
    success = response.get("success", False)
    duration = response.get("duration", 0)
    
    status_icon = "✅" if success else "❌"
    adversarial_icon = "🎭" if payload.get("enable_adversarial") else ""
    economic_icon = "💰" if payload.get("economic_attack") else ""
    
    print(f"{status_icon} [{index+1:3d}/{N}] {profile} | {anomaly} | {phase} | {duration:.2f}s {adversarial_icon}{economic_icon}")
    
    if VERBOSE and response.get("lambda_response"):
        lambda_resp = response["lambda_response"]
        status_code = lambda_resp.get("statusCode", "unknown")
        print(f"         └─ Lambda Status: {status_code}")
        
        # Try to parse body for additional info
        try:
            body = json.loads(lambda_resp.get("body", "{}"))
            if "anomaly_detected" in body:
                anomaly_detected = body["anomaly_detected"]
                detection_icon = "🚨" if anomaly_detected else "🟢"
                print(f"         └─ Anomaly Detected: {detection_icon} {anomaly_detected}")
        except:
            pass

def print_execution_summary(start_time: float, responses: List[Dict]):
    """Print execution summary"""
    end_time = time.time()
    total_duration = end_time - start_time
    
    successful = sum(1 for r in responses if r.get("success", False))
    failed = len(responses) - successful
    
    avg_duration = sum(r.get("duration", 0) for r in responses) / len(responses) if responses else 0
    min_duration = min(r.get("duration", 0) for r in responses) if responses else 0
    max_duration = max(r.get("duration", 0) for r in responses) if responses else 0
    
    print(f"\n📊 Execution Summary")
    print(f"{'='*50}")
    print(f"🎯 Total Invocations: {N}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {successful/N*100:.1f}%")
    print(f"⏱️  Total Time: {total_duration:.2f}s")
    print(f"⚡ Avg Response Time: {avg_duration:.2f}s")
    print(f"🏃 Min Response Time: {min_duration:.2f}s")
    print(f"🐌 Max Response Time: {max_duration:.2f}s")
    
    # Anomaly breakdown
    anomaly_counts = {}
    for i, response in enumerate(responses):
        # We need to map back to payloads - this is simplified
        pass  # Could be enhanced with more detailed tracking
    
    print(f"\n📁 Output Files:")
    print(f"├── 📋 Payloads: {OUTPUT_DIR}/payloads/")
    print(f"├── 📤 Responses: {OUTPUT_DIR}/responses/")
    print(f"└── 📃 Master Log: {OUTPUT_DIR}/invocation_master_log.jsonl")

def process_batch(batch_payloads: List[Dict], batch_start_index: int) -> List[Dict]:
    """Process a batch of payloads"""
    print(f"\n🔄 Processing batch {batch_start_index//BATCH_SIZE + 1} ({len(batch_payloads)} invocations)")
    
    responses = []
    for i, payload in enumerate(batch_payloads):
        actual_index = batch_start_index + i
        
        # Save payload
        save_payload(payload, actual_index)
        
        # Invoke function
        response = invoke_lambda_function(payload, actual_index)
        
        # Save response
        save_response(response, actual_index)
        
        # Update logs
        update_master_log(payload, response)
        
        # Print summary
        print_invocation_summary(payload, response, actual_index)
        
        responses.append(response)
        
        # Delay between invocations (except last one)
        if i < len(batch_payloads) - 1:
            time.sleep(DELAY)
    
    return responses

def main():
    """Main execution function"""
    print(f"🚀 SCAFAD Layer 0 Enhanced Invocation Script")
    print(f"{'='*60}")
    print(f"📊 Configuration:")
    print(f"   • Invocations: {N}")
    print(f"   • Mode: {MODE}")
    print(f"   • Adversarial: {'✅' if ENABLE_ADVERSARIAL else '❌'}")
    print(f"   • Economic Attacks: {'✅' if ENABLE_ECONOMIC else '❌'}")
    print(f"   • Function: {FUNCTION_NAME}")
    print(f"   • Delay: {DELAY}s")
    print(f"   • Output: {OUTPUT_DIR}/")
    print(f"   • Seed: {SEED}")
    
    if BATCH_SIZE > 0:
        print(f"   • Batch Size: {BATCH_SIZE}")
    
    # Create output directories
    create_output_directories()
    
    # Generate all payloads
    print(f"\n📦 Generating {N} enhanced payloads...")
    payloads = []
    for i in range(N):
        payload = generate_enhanced_payload(i)
        payloads.append(payload)
    
    print(f"✅ Generated {len(payloads)} payloads")
    
    # Execute invocations
    start_time = time.time()
    print(f"\n🚀 Starting invocations...")
    
    all_responses = []
    
    if BATCH_SIZE > 0:
        # Process in batches
        for batch_start in range(0, N, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, N)
            batch_payloads = payloads[batch_start:batch_end]
            
            batch_responses = process_batch(batch_payloads, batch_start)
            all_responses.extend(batch_responses)
            
            # Pause between batches
            if batch_end < N:
                print(f"⏸️  Batch complete, pausing {DELAY*2:.1f}s before next batch...")
                time.sleep(DELAY * 2)
    else:
        # Process all at once
        all_responses = process_batch(payloads, 0)
    
    # Print final summary
    print_execution_summary(start_time, all_responses)
    
    print(f"\n🎉 All {N} invocations completed!")
    print(f"💡 Next steps:")
    print(f"   1. Check logs: python fetch_logs.py")
    print(f"   2. Analyze results in: {OUTPUT_DIR}/")
    print(f"   3. Review master log: {OUTPUT_DIR}/invocation_master_log.jsonl")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  Invocation interrupted by user")
    except Exception as e:
        print(f"\n❌ Invocation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)