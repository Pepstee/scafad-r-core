#!/usr/bin/env python3
"""
Debug SCAFAD invoke failures
"""

import json
import os
import glob

def check_response_files():
    """Check the actual response files to see what's failing"""
    print("Debugging SCAFAD invoke failures...")
    print("=" * 50)
    
    # Check if telemetry directory exists
    if not os.path.exists("telemetry"):
        print("❌ No telemetry directory found")
        return
    
    # Check response files
    response_files = glob.glob("telemetry/responses/*.json")
    
    if not response_files:
        print("❌ No response files found")
        return
    
    print(f"📋 Found {len(response_files)} response files")
    print()
    
    # Analyze the most recent responses
    for i, response_file in enumerate(sorted(response_files)[-3:]):  # Last 3 files
        print(f"📄 Response {i+1}: {os.path.basename(response_file)}")
        
        try:
            with open(response_file, 'r', encoding='utf-8') as f:
                response = json.load(f)
            
            # Extract key information
            success = response.get("success", False)
            duration = response.get("duration", 0)
            error = response.get("error", "No error info")
            error_type = response.get("error_type", "Unknown")
            
            print(f"   Success: {success}")
            print(f"   Duration: {duration:.2f}s")
            
            if not success:
                print(f"   Error Type: {error_type}")
                print(f"   Error: {error}")
                
                # Check for specific error patterns
                if "sam" in error.lower():
                    print("   🔍 DIAGNOSIS: Still trying to use SAM local")
                elif "boto3" in error.lower() or "lambda" in error.lower():
                    print("   🔍 DIAGNOSIS: AWS Lambda connection issue")
                elif "function" in error.lower() and "not" in error.lower():
                    print("   🔍 DIAGNOSIS: Function name or permission issue")
                else:
                    print("   🔍 DIAGNOSIS: Unknown error")
            
            # Check if there's stdout/stderr info
            if "stdout" in response:
                stdout = response["stdout"]
                stderr = response["stderr"]
                
                if stderr:
                    print(f"   Stderr: {stderr[:200]}...")  # First 200 chars
                
                if stdout:
                    print(f"   Stdout: {stdout[:200]}...")  # First 200 chars
            
            print()
            
        except Exception as e:
            print(f"   ❌ Could not read response file: {e}")
            print()

def check_master_log():
    """Check the master log for additional insights"""
    print("📋 Checking master log...")
    
    master_log_path = "telemetry/invocation_master_log.jsonl"
    
    if not os.path.exists(master_log_path):
        print("❌ No master log found")
        return
    
    try:
        with open(master_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("❌ Master log is empty")
            return
        
        print(f"📊 Found {len(lines)} log entries")
        
        # Check the last few entries
        for i, line in enumerate(lines[-3:]):  # Last 3 entries
            try:
                entry = json.loads(line)
                
                payload_summary = entry.get("payload_summary", {})
                response_summary = entry.get("response_summary", {})
                
                anomaly = payload_summary.get("anomaly", "unknown")
                success = response_summary.get("success", False)
                error_type = response_summary.get("error_type", "none")
                
                print(f"   Entry {i+1}: {anomaly} -> {'✅' if success else '❌'} ({error_type})")
                
            except json.JSONDecodeError:
                print(f"   Entry {i+1}: Invalid JSON")
        
        print()
        
    except Exception as e:
        print(f"❌ Could not read master log: {e}")

def check_current_invoke_method():
    """Check what invoke method is currently being used"""
    print("🔍 Checking current invoke method...")
    
    if not os.path.exists("invoke.py"):
        print("❌ invoke.py not found")
        return
    
    with open("invoke.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for invoke method indicators
    if "sam local invoke" in content:
        print("❌ Still using SAM local invoke")
        print("   Need to run: python fix_invoke_method.py")
    elif "boto3" in content and "lambda_client.invoke" in content:
        print("✅ Using direct AWS Lambda calls")
    elif "subprocess.run" in content and "sam" in content:
        print("❌ Using subprocess to call SAM")
        print("   Need to run: python fix_invoke_method.py")
    else:
        print("❓ Unknown invoke method")
    
    print()

def suggest_fixes():
    """Suggest fixes based on the analysis"""
    print("🔧 SUGGESTED FIXES:")
    print("=" * 30)
    
    print("1. If still using SAM local:")
    print("   python fix_invoke_method.py")
    print()
    
    print("2. Test direct Lambda call:")
    print("   python simple_test.py")
    print()
    
    print("3. Check AWS credentials:")
    print("   aws sts get-caller-identity")
    print()
    
    print("4. Verify function exists:")
    print("   aws lambda get-function --function-name scafad-test-stack-HelloWorldFunction-k79tX3iBcK74")
    print()
    
    print("5. Manual test with curl/awscli:")
    print("   aws lambda invoke --function-name scafad-test-stack-HelloWorldFunction-k79tX3iBcK74 --payload '{\"test\":true}' response.json")

def main():
    """Main debug function"""
    print("🔍 SCAFAD Invoke Failure Debugger")
    print("=" * 50)
    
    check_current_invoke_method()
    check_response_files()
    check_master_log()
    suggest_fixes()

if __name__ == "__main__":
    main()