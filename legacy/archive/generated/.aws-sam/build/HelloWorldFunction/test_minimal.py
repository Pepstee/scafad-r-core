#!/usr/bin/env python3
"""
Debug version of SCAFAD invoke script
Focuses on diagnosing SAM local invoke issues
"""

import json
import subprocess
import sys
import os
from datetime import datetime

def check_prerequisites():
    """Check if all prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check SAM CLI
    try:
        result = subprocess.run(['sam', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ SAM CLI: {result.stdout.strip()}")
        else:
            print("❌ SAM CLI not found or not working")
            return False
    except FileNotFoundError:
        print("❌ SAM CLI not installed")
        return False
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker not found or not working")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False
    
    # Check if Docker daemon is running
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
        else:
            print("❌ Docker daemon not running")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker daemon check failed: {e}")
        return False
    
    return True

def validate_template():
    """Validate SAM template"""
    print("\n🔍 Validating SAM template...")
    
    if not os.path.exists('template.yaml'):
        print("❌ template.yaml not found")
        return False
    
    try:
        result = subprocess.run(['sam', 'validate'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Template is valid")
            return True
        else:
            print("❌ Template validation failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Template validation error: {e}")
        return False

def test_basic_invoke():
    """Test basic Lambda invoke with minimal payload"""
    print("\n🧪 Testing basic Lambda invoke...")
    
    # Create minimal test payload
    test_payload = {
        "test": True,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("test_payload.json", "w") as f:
        json.dump(test_payload, f, indent=2)
    
    # Try to invoke
    cmd = ["sam", "local", "invoke", "HelloWorldFunction", "--event", "test_payload.json"]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"\n📊 Results:")
        print(f"   Return code: {result.returncode}")
        print(f"   Success: {'✅' if result.returncode == 0 else '❌'}")
        
        if result.stdout:
            print(f"\n📝 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print(f"\n⚠️ STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def test_with_detailed_logging():
    """Test with detailed logging enabled"""
    print("\n🔍 Testing with detailed logging...")
    
    # Set environment variables for verbose output
    env = os.environ.copy()
    env['SAM_CLI_DEBUG'] = '1'
    
    cmd = ["sam", "local", "invoke", "HelloWorldFunction", 
           "--event", "test_payload.json", "--debug"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              timeout=120, env=env)
        
        print(f"📊 Detailed results:")
        print(f"   Return code: {result.returncode}")
        
        if result.stdout:
            print(f"\n📝 Detailed STDOUT:")
            print(result.stdout[-2000:])  # Last 2000 chars to avoid overflow
        
        if result.stderr:
            print(f"\n⚠️ Detailed STDERR:")
            print(result.stderr[-2000:])  # Last 2000 chars to avoid overflow
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Detailed test failed: {e}")
        return False

def analyze_logs():
    """Analyze any log files created"""
    print("\n📁 Checking for log files...")
    
    log_files = [f for f in os.listdir('.') if f.endswith('.log') or 'sam' in f.lower()]
    
    if log_files:
        print(f"📄 Found log files: {log_files}")
        for log_file in log_files[:3]:  # Limit to first 3 files
            if os.path.getsize(log_file) < 10000:  # Only show small files
                print(f"\n📄 Contents of {log_file}:")
                try:
                    with open(log_file, 'r') as f:
                        print(f.read())
                except Exception as e:
                    print(f"   Could not read {log_file}: {e}")
    else:
        print("📄 No log files found")

def main():
    """Main diagnostic function"""
    print("🚀 SCAFAD Lambda Diagnostic Tool")
    print("=" * 50)
    
    # Run all checks
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        return False
    
    if not validate_template():
        print("\n❌ Template validation failed. Please fix the template.")
        return False
    
    # Try basic invoke
    basic_success = test_basic_invoke()
    
    if not basic_success:
        print("\n🔍 Basic invoke failed, trying with detailed logging...")
        test_with_detailed_logging()
    
    analyze_logs()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if basic_success:
        print("✅ Basic Lambda invoke working!")
        print("💡 The issue might be with your SCAFAD payload complexity.")
        print("   Try running your original invoke.py with --verbose flag.")
    else:
        print("❌ Basic Lambda invoke failed.")
        print("💡 Common solutions:")
        print("   1. Restart Docker Desktop")
        print("   2. Run: sam build --use-container")
        print("   3. Check if you have enough disk space")
        print("   4. Try: docker system prune")
        print("   5. Verify Python version compatibility")
    
    return basic_success

if __name__ == "__main__":
    main()