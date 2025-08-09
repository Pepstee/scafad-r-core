#!/usr/bin/env python3
"""
Fix SCAFAD invoke.py to use direct Lambda calls instead of SAM local
"""

import re
import os

def fix_invoke_method():
    """Fix invoke.py to use direct Lambda calls instead of SAM local"""
    
    print("Fixing invoke method in invoke.py...")
    
    if not os.path.exists("invoke.py"):
        print("   invoke.py not found!")
        return False
    
    # Read current content
    with open("invoke.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup current version
    with open("invoke.py.method_backup", 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Replace the invoke_lambda_function to use direct AWS calls
    new_invoke_function = '''def invoke_lambda_function(payload: Dict, index: int) -> Dict:
    """Invoke the Lambda function using direct AWS Lambda calls (not SAM local)"""
    
    start_time = time.time()
    
    try:
        # Use boto3 to call the deployed Lambda function directly
        import boto3
        
        lambda_client = boto3.client('lambda')
        
        if VERBOSE:
            print(f"   [ROCKET] Invoking AWS Lambda function: {FUNCTION_NAME}")
        
        # Convert payload to JSON string
        payload_json = json.dumps(payload, ensure_ascii=False)
        
        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName=FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=payload_json
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse the response
        response_payload = response['Payload'].read()
        
        # Try to parse as JSON
        try:
            lambda_response = json.loads(response_payload)
        except json.JSONDecodeError:
            lambda_response = {"raw_response": response_payload.decode('utf-8')}
        
        # Check for function errors
        function_error = response.get('FunctionError')
        success = function_error is None and lambda_response.get('statusCode') == 200
        
        # Build response object
        result = {
            "invocation_index": index,
            "success": success,
            "duration": duration,
            "returncode": 0 if success else 1,
            "lambda_response": lambda_response,
            "aws_response_metadata": {
                "status_code": response['ResponseMetadata']['HTTPStatusCode'],
                "request_id": response['ResponseMetadata']['RequestId'],
                "function_error": function_error
            },
            "timestamp": datetime.now().isoformat(),
            "payload_id": payload.get("payload_id", f"aws_{index}")
        }
        
        # Extract SCAFAD-specific metrics if available
        if isinstance(lambda_response.get("body"), str):
            try:
                body = json.loads(lambda_response["body"])
                result["scafad_metrics"] = {
                    "anomaly_detected": body.get("anomaly_detected"),
                    "telemetry_id": body.get("telemetry_id"),
                    "processing_time_ms": body.get("processing_time_ms"),
                    "economic_risk_score": body.get("economic_risk_score"),
                    "completeness_score": body.get("completeness_score")
                }
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(lambda_response.get("body"), dict):
            body = lambda_response["body"]
            result["scafad_metrics"] = {
                "anomaly_detected": body.get("anomaly_detected"),
                "telemetry_id": body.get("telemetry_id"),
                "processing_time_ms": body.get("processing_time_ms"),
                "economic_risk_score": body.get("economic_risk_score"),
                "completeness_score": body.get("completeness_score")
            }
        
        return result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "invocation_index": index,
            "success": False,
            "duration": duration,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
            "payload_id": payload.get("payload_id", f"error_{index}")
        }'''
    
    # Find the existing invoke_lambda_function and replace it
    # Look for the function definition
    pattern = r'def invoke_lambda_function\(.*?\n(?:.*\n)*?(?=def|\Z)'
    
    if re.search(pattern, content, re.MULTILINE):
        # Replace the function
        new_content = re.sub(pattern, new_invoke_function + '\n\n', content, flags=re.MULTILINE)
        
        # Write the updated content
        with open("invoke.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("   ✓ Updated invoke_lambda_function to use direct AWS Lambda calls")
        return True
    else:
        print("   ❌ Could not find invoke_lambda_function to replace")
        return False

def add_boto3_import():
    """Ensure boto3 is imported at the top of the file"""
    
    with open("invoke.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if boto3 is already imported
    if "import boto3" not in content:
        # Find the import section and add boto3
        lines = content.split('\n')
        import_index = -1
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_index = i
        
        if import_index >= 0:
            # Insert boto3 import after the last import
            lines.insert(import_index + 1, "import boto3")
            
            content = '\n'.join(lines)
            
            with open("invoke.py", 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("   ✓ Added boto3 import")
        else:
            print("   ⚠️ Could not find import section to add boto3")
    else:
        print("   ✓ boto3 already imported")

def main():
    """Apply the invoke method fix"""
    
    print("SCAFAD Invoke Method Fix")
    print("=" * 30)
    print("Switching from SAM local to direct AWS Lambda calls...")
    print()
    
    # Apply fixes
    success = fix_invoke_method()
    add_boto3_import()
    
    if success:
        print("\n✓ INVOKE METHOD FIXED")
        print("=" * 30)
        print("Changes made:")
        print("1. ✓ Using direct AWS Lambda calls instead of SAM local")
        print("2. ✓ Added boto3 import")
        print("3. ✓ Backup saved as invoke.py.method_backup")
        
        print("\n[NEXT STEPS]")
        print("1. Test: python invoke.py --n 3 --mode test")
        print("2. Should see 100% success rate now!")
        print("3. The function will call your deployed Lambda directly")
    else:
        print("\n❌ COULD NOT APPLY FIX")
        print("Manual fix needed:")
        print("1. Open invoke.py in a text editor")
        print("2. Find the invoke_lambda_function")
        print("3. Replace SAM local calls with boto3 Lambda calls")

if __name__ == "__main__":
    main()