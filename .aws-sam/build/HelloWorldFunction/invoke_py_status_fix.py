# Fix for invoke.py - change the success check in invoke_lambda_function()

# OLD CODE (around line 250):
# response = {
#     "invocation_index": index,
#     "success": result.returncode == 0,
#     ...
# }

# NEW CODE - Add this function to check Lambda response status:
def is_successful_response(result, lambda_response):
    """Check if Lambda response indicates success"""
    # First check if subprocess succeeded
    if result.returncode != 0:
        return False
    
    # Then check Lambda status code - accept both 200 and 202
    if lambda_response and 'statusCode' in lambda_response:
        return lambda_response['statusCode'] in [200, 202]
    
    # If no status code, consider it success if subprocess succeeded
    return True

# UPDATED response creation:
# Parse response with enhanced error handling
response = {
    "invocation_index": index,
    "success": False,  # Will be updated below
    "duration": duration,
    "returncode": result.returncode,
    "stdout": result.stdout,
    "stderr": result.stderr,
    "timestamp": datetime.now().isoformat(),
    "payload_id": payload.get("payload_id", f"unknown_{index}")
}

# Enhanced JSON response extraction
try:
    stdout_lines = result.stdout.split('\n')
    json_response = None
    
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
        
        # UPDATE SUCCESS CHECK HERE:
        response["success"] = is_successful_response(result, json_response)
        
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
    else:
        # If no JSON response but subprocess succeeded, still consider success
        response["success"] = result.returncode == 0

except Exception as e:
    if VERBOSE:
        print(f"   ⚠️  Error parsing Lambda response: {e}")
    # Fallback: success based on return code only
    response["success"] = result.returncode == 0