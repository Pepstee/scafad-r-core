#!/usr/bin/env python3
"""
SAM Local Invoke Diagnostic Tool
Helps identify why SAM local invocations are failing
"""

import subprocess
import json
import os
import sys
from typing import Dict, List

def check_docker():
    """Check if Docker is running and accessible"""
    print("🐳 Checking Docker...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker version: {result.stdout.strip()}")
        else:
            print(f"❌ Docker version check failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker not found or not running: {e}")
        return False
    
    try:
        result = subprocess.run(['docker', 'ps'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
            return True
        else:
            print(f"❌ Docker daemon not accessible: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker daemon check failed: {e}")
        return False

def check_sam_cli():
    """Check SAM CLI installation"""
    print("\n🔨 Checking SAM CLI...")
    
    try:
        result = subprocess.run(['sam', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ SAM CLI version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ SAM CLI check failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ SAM CLI not found: {e}")
        return False

def check_sam_template():
    """Check SAM template.yaml"""
    print("\n📄 Checking SAM template...")
    
    if not os.path.exists('template.yaml'):
        print("❌ template.yaml not found")
        return False
    
    try:
        with open('template.yaml', 'r') as f:
            content = f.read()
        
        required_sections = ['AWSTemplateFormatVersion', 'Transform', 'Resources']
        missing = [s for s in required_sections if s not in content]
        
        if missing:
            print(f"❌ Missing sections in template.yaml: {missing}")
            return False
        else:
            print("✅ template.yaml structure looks good")
            return True
            
    except Exception as e:
        print(f"❌ Error reading template.yaml: {e}")
        return False

def check_build_status():
    """Check if SAM application is built"""
    print("\n🏗️  Checking build status...")
    
    if os.path.exists('.aws-sam/build'):
        print("✅ .aws-sam/build directory exists")
        if os.path.exists('.aws-sam/build/template.yaml'):
            print("✅ Built template.yaml exists")
            return True
        else:
            print("❌ Built template.yaml missing")
            return False
    else:
        print("❌ .aws-sam/build directory not found")
        print("💡 Run 'sam build' to build the application")
        return False

def test_simple_invoke():
    """Test a simple SAM local invoke"""
    print("\n🧪 Testing simple SAM invoke...")
    
    # Create a minimal test event
    test_event = {
        "test": True,
        "message": "Hello SAM"
    }
    
    try:
        with open('test_event.json', 'w') as f:
            json.dump(test_event, f, indent=2)
        
        print("📝 Created test_event.json")
        
        # Try the invoke
        cmd = ['sam', 'local', 'invoke', 'HelloWorldFunction', '--event', 'test_event.json']
        print(f"🚀 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print("✅ Simple invoke succeeded!")
            return True
        else:
            print("❌ Simple invoke failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ SAM invoke timed out")
        return False
    except Exception as e:
        print(f"❌ Error during test invoke: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists('test_event.json'):
            os.remove('test_event.json')

def check_function_code():
    """Basic check of function code"""
    print("\n🐍 Checking function code...")
    
    if not os.path.exists('app.py'):
        print("❌ app.py not found")
        return False
    
    try:
        # Try to import and basic syntax check
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Check for lambda_handler
        if 'def lambda_handler' not in content:
            print("❌ lambda_handler function not found in app.py")
            return False
        
        print("✅ app.py exists and has lambda_handler")
        
        # Try basic syntax check
        compile(content, 'app.py', 'exec')
        print("✅ app.py syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking app.py: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("🔍 SAM Local Invoke Diagnostic Tool")
    print("=" * 50)
    
    checks = [
        ("Docker", check_docker),
        ("SAM CLI", check_sam_cli),
        ("SAM Template", check_sam_template),
        ("Build Status", check_build_status),
        ("Function Code", check_function_code),
        ("Simple Invoke", test_simple_invoke)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("📊 Diagnostic Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:15} : {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your SAM setup looks good.")
    else:
        print("\n💡 Recommendations:")
        if not results.get("Docker", True):
            print("   - Install and start Docker Desktop")
        if not results.get("SAM CLI", True):
            print("   - Install AWS SAM CLI")
        if not results.get("Build Status", True):
            print("   - Run 'sam build' to build your application")
        if not results.get("Function Code", True):
            print("   - Fix syntax errors in app.py")
        if not results.get("Simple Invoke", True):
            print("   - Check the error output above for specific issues")

if __name__ == "__main__":
    main()