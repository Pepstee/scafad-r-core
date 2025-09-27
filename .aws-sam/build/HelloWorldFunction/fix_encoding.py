#!/usr/bin/env python3
"""
File Encoding Cleanup Tool
Fixes UTF-8 encoding issues in project files
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the original file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"📋 Backed up {filepath} to {backup_path}")
    return backup_path

def fix_file_encoding(filepath, target_encoding='utf-8'):
    """Fix file encoding issues"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"🔧 Fixing encoding for: {filepath}")
    
    # Backup original
    backup_path = backup_file(filepath)
    
    # Try different encodings to read the file
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    content = None
    source_encoding = None
    
    for encoding in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            source_encoding = encoding
            print(f"✅ Successfully read with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"❌ Failed to read with {encoding} encoding")
            continue
        except Exception as e:
            print(f"❌ Error with {encoding}: {e}")
            continue
    
    if content is None:
        print(f"❌ Could not read {filepath} with any encoding")
        return False
    
    # Clean the content - remove any problematic characters
    # Replace common problematic characters
    replacements = {
        '\x8f': '',  # Remove problematic byte
        '\xff': '',  # Remove problematic byte
        '\ufeff': '',  # Remove BOM
        '\r\n': '\n',  # Normalize line endings
        '\r': '\n'     # Normalize line endings
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write back with clean UTF-8 encoding
    try:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
        print(f"✅ Successfully wrote {filepath} with UTF-8 encoding")
        return True
    except Exception as e:
        print(f"❌ Error writing {filepath}: {e}")
        # Restore backup if write failed
        shutil.copy2(backup_path, filepath)
        print(f"🔄 Restored original file from backup")
        return False

def fix_json_file(filepath):
    """Fix and validate JSON file"""
    import json
    
    print(f"🔧 Fixing JSON file: {filepath}")
    
    # First fix encoding
    if not fix_file_encoding(filepath):
        return False
    
    # Then validate and clean JSON
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Write back with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON file {filepath} is now valid and clean")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON validation failed for {filepath}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing JSON {filepath}: {e}")
        return False

def main():
    """Fix encoding issues in project files"""
    print("🔧 SCAFAD File Encoding Cleanup Tool")
    print("=" * 50)
    
    files_to_fix = [
        ('app.py', 'python'),
        ('event.json', 'json'),
        ('payload.json', 'json'),
        ('template.yaml', 'text'),
        ('requirements.txt', 'text'),
        ('samconfig.toml', 'text')
    ]
    
    results = {}
    
    for filepath, filetype in files_to_fix:
        if os.path.exists(filepath):
            print(f"\n🔍 Processing {filepath} ({filetype})...")
            
            if filetype == 'json':
                results[filepath] = fix_json_file(filepath)
            else:
                results[filepath] = fix_file_encoding(filepath)
        else:
            print(f"⏭️  Skipping {filepath} (not found)")
            results[filepath] = None
    
    print("\n" + "=" * 50)
    print("📊 Encoding Fix Summary")
    print("=" * 50)
    
    for filepath, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ FIXED"
        else:
            status = "❌ FAILED"
        
        print(f"{filepath:20} : {status}")
    
    # Create a clean event.json if it doesn't exist or failed
    if not os.path.exists('event.json') or not results.get('event.json', False):
        print(f"\n📝 Creating clean event.json...")
        clean_event = {
            "test_mode": True,
            "anomaly": "benign",
            "function_profile_id": "func_A",
            "execution_phase": "invoke",
            "concurrency_id": "TST",
            "force_starvation": False,
            "invocation_timestamp": 1705158400.0,
            "enable_adversarial": False,
            "attack_type": "adaptive",
            "payload_id": "test_001",
            "layer0_enabled": True,
            "schema_version": "v4.2"
        }
        
        try:
            with open('event.json', 'w', encoding='utf-8') as f:
                json.dump(clean_event, f, indent=2)
            print("✅ Created clean event.json")
        except Exception as e:
            print(f"❌ Failed to create event.json: {e}")
    
    fixed_count = sum(1 for r in results.values() if r is True)
    total_count = sum(1 for r in results.values() if r is not None)
    
    print(f"\n🎯 Summary: Fixed {fixed_count}/{total_count} files")
    
    if fixed_count > 0:
        print("\n💡 Next steps:")
        print("   1. Try: sam local invoke HelloWorldFunction --event event.json")
        print("   2. If successful, run: python invoke.py --n 1 -v")
        print("   3. Monitor the output for any remaining issues")

if __name__ == "__main__":
    main()