#!/usr/bin/env python3
"""
SAM CLI Path Fix Script
Resolves SAM CLI path issues for subprocess calls
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def find_sam_executable():
    """Find SAM CLI executable in various locations"""
    print("🔍 Searching for SAM CLI executable...")
    
    # Common locations for SAM CLI on Windows
    possible_paths = [
        r"C:\Program Files\Amazon\AWSSAMCLI\sam.exe",
        r"C:\Program Files\Amazon\AWSSAMCLI\bin\sam.exe",
        r"C:\Program Files (x86)\Amazon\AWSSAMCLI\sam.exe",
        r"C:\Users\{username}\AppData\Local\aws-sam-cli\sam.exe".format(username=os.getenv('USERNAME')),
        r"C:\Python\Scripts\sam.exe",
        r"C:\Python39\Scripts\sam.exe",
        r"C:\Python310\Scripts\sam.exe",
        r"C:\Python311\Scripts\sam.exe",
    ]
    
    # Check PATH first
    sam_path = shutil.which('sam')
    if sam_path:
        print(f"✅ Found SAM in PATH: {sam_path}")
        return sam_path
    
    # Check common installation paths
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found SAM at: {path}")
            return path
    
    # Check if sam.cmd exists (Windows batch file)
    sam_cmd = shutil.which('sam.cmd')
    if sam_cmd:
        print(f"✅ Found SAM CMD: {sam_cmd}")
        return sam_cmd
    
    print("❌ SAM CLI not found in common locations")
    return None

def test_sam_executable(sam_path):
    """Test if SAM executable works"""
    print(f"🧪 Testing SAM executable: {sam_path}")
    
    try:
        result = subprocess.run([sam_path, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ SAM works: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ SAM test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ SAM test error: {e}")
        return False

def create_sam_wrapper():
    """Create a Python wrapper for SAM CLI calls"""
    print("📝 Creating SAM wrapper for reliable subprocess calls...")
    
    sam_path = find_sam_executable()
    if not sam_path:
        print("❌ Cannot create wrapper - SAM CLI not found")
        return False
    
    if not test_sam_executable(sam_path):
        print("❌ Cannot create wrapper - SAM CLI not working")
        return False
    
    wrapper_content = f'''#!/usr/bin/env python3
"""
SAM CLI Wrapper for SCAFAD
Auto-generated wrapper to handle SAM CLI path issues
"""

import subprocess
import sys
import os

# SAM CLI executable path
SAM_PATH = r"{sam_path}"

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
        print(f"SAM Path: {{SAM_PATH}}")
'''
    
    try:
        with open('sam_wrapper.py', 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        print("✅ Created sam_wrapper.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create wrapper: {e}")
        return False

def update_invoke_script():
    """Update invoke.py to use the SAM wrapper"""
    print("🔧 Updating invoke.py to use SAM wrapper...")
    
    if not os.path.exists('invoke.py'):
        print("❌ invoke.py not found")
        return False
    
    try:
        # Read current invoke.py
        with open('invoke.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace SAM command construction
        old_cmd = 'cmd = ["sam", "local", "invoke", FUNCTION_NAME, "--event", "payload.json"]'
        new_cmd = '''# Use SAM wrapper for reliable path resolution
        try:
            from sam_wrapper import invoke_function
            # Use wrapper function
            result = invoke_function(FUNCTION_NAME, "payload.json", 
                                   capture_output=True, text=True, timeout=120)
        except ImportError:
            # Fallback to direct SAM call
            cmd = ["sam", "local", "invoke", FUNCTION_NAME, "--event", "payload.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)'''
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
            
            # Write back the updated file
            with open('invoke.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Updated invoke.py to use SAM wrapper")
            return True
        else:
            print("⚠️  Could not find expected command pattern in invoke.py")
            return False
            
    except Exception as e:
        print(f"❌ Error updating invoke.py: {e}")
        return False

def main():
    """Main path fix function"""
    print("🔧 SAM CLI Path Fix Tool")
    print("=" * 40)
    
    steps = [
        ("Find SAM CLI", find_sam_executable),
        ("Create SAM Wrapper", create_sam_wrapper),
        ("Update Invoke Script", update_invoke_script)
    ]
    
    sam_path = None
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if step_name == "Find SAM CLI":
            sam_path = step_func()
            success = sam_path is not None
        else:
            success = step_func()
        
        if success:
            print(f"✅ {step_name} completed")
        else:
            print(f"❌ {step_name} failed")
            break
    
    print("\n" + "=" * 40)
    if sam_path:
        print(f"🎯 SAM CLI found at: {sam_path}")
        print("✅ Path fix completed!")
        print("\n💡 Next steps:")
        print("   1. Run: python fix_encoding.py")
        print("   2. Test: sam local invoke HelloWorldFunction --event event.json")
        print("   3. Try: python invoke.py --n 1 -v")
    else:
        print("❌ SAM CLI path fix failed")
        print("💡 Manual installation required:")
        print("   1. Download SAM CLI from: https://aws.amazon.com/serverless/sam/")
        print("   2. Install and add to PATH")
        print("   3. Restart command prompt")

if __name__ == "__main__":
    main()