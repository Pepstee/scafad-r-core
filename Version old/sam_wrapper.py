#!/usr/bin/env python3
"""
SAM CLI Wrapper for SCAFAD
Auto-generated wrapper to handle SAM CLI path issues
"""

import subprocess
import sys
import os

# SAM CLI executable path
SAM_PATH = r"C:\Program Files\Amazon\AWSSAMCLI\bin\sam.CMD"

def run_sam_command(args, **kwargs):
    """Run SAM command with proper path resolution"""
    cmd = [SAM_PATH] + args
    return subprocess.run(cmd, **kwargs)

def invoke_function(function_name, event_file, **kwargs):
    """Invoke Lambda function using SAM CLI"""
    args = ["local", "invoke", function_name, "--event", event_file]
    return run_sam_command(args, **kwargs)

if __name__ == "__main__":
    # Pass through command line arguments
    if len(sys.argv) > 1:
        result = run_sam_command(sys.argv[1:])
        sys.exit(result.returncode)
    else:
        print("SAM CLI Wrapper - Use with SAM commands")
        print(f"SAM Path: {SAM_PATH}")
