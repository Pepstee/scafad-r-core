#!/usr/bin/env python3
"""
SCAFAD Layer 0 Cleanup Script - Fixed for Windows
================================================

This script will move unnecessary, duplicate, and fragmented files to a trash folder
while preserving the essential components needed for your dissertation.
"""

import os
import shutil
import json
import glob
from datetime import datetime
from pathlib import Path
import logging

# Configuration - Fixed paths for Windows
SOURCE_DIR = "C:/Users/Gutua/OneDrive/Documents/scafad-lambda"
TRASH_DIR = "C:/Users/Gutua/OneDrive/Documents/scafad-layer0-trash"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_log.txt'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_trash_structure():
    """Create organized trash directory structure"""
    trash_subdirs = [
        "fragmented_files",      # Incomplete/fragmented files
        "duplicate_logic",       # Files with duplicate functionality
        "analysis_outputs",      # Generated analysis files
        "temp_files",           # Temporary files
        "old_versions",         # Old version files
        "backup_originals"      # Backup of original files before cleanup
    ]
    
    os.makedirs(TRASH_DIR, exist_ok=True)
    
    for subdir in trash_subdirs:
        os.makedirs(os.path.join(TRASH_DIR, subdir), exist_ok=True)
    
    logger.info(f"Created trash directory structure at: {TRASH_DIR}")

def backup_current_state():
    """Create a complete backup before cleanup"""
    backup_dir = os.path.join(TRASH_DIR, "backup_originals", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    try:
        shutil.copytree(SOURCE_DIR, backup_dir)
        logger.info(f"✅ Complete backup created at: {backup_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return False

def analyze_files():
    """Analyze current files to identify cleanup targets"""
    if not os.path.exists(SOURCE_DIR):
        logger.error(f"Source directory not found: {SOURCE_DIR}")
        return {}
    
    analysis = {
        "fragmented_files": [],
        "duplicate_logic": [],
        "analysis_outputs": [],
        "temp_files": [],
        "total_files": 0,
        "files_to_move": 0
    }
    
    # Get all files in source directory
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, SOURCE_DIR)
            analysis["total_files"] += 1
            
            # Identify fragmented/incomplete files
            if is_fragmented_file(full_path, file):
                analysis["fragmented_files"].append(rel_path)
            
            # Identify duplicate logic
            elif is_duplicate_logic(full_path, file):
                analysis["duplicate_logic"].append(rel_path)
            
            # Identify analysis outputs
            elif is_analysis_output(full_path, file):
                analysis["analysis_outputs"].append(rel_path)
            
            # Identify temp files
            elif is_temp_file(full_path, file):
                analysis["temp_files"].append(rel_path)
    
    analysis["files_to_move"] = (
        len(analysis["fragmented_files"]) +
        len(analysis["duplicate_logic"]) +
        len(analysis["analysis_outputs"]) +
        len(analysis["temp_files"])
    )
    
    return analysis

def is_fragmented_file(full_path, filename):
    """Identify fragmented or incomplete files"""
    fragmented_indicators = [
        lambda f, p: f.endswith('.py') and is_incomplete_python_file(p),
        lambda f, p: 'incomplete' in f.lower(),
        lambda f, p: 'fragment' in f.lower(),
        lambda f, p: f.startswith('temp_') or f.startswith('tmp_'),
    ]
    
    return any(indicator(filename, full_path) for indicator in fragmented_indicators)

def is_incomplete_python_file(filepath):
    """Check if a Python file appears incomplete based on content"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Signs of incomplete files
        incomplete_signs = [
            len(content.strip()) < 50,  # Very short files
            content.count('"""') % 2 != 0,  # Unmatched docstrings
            content.endswith('...') or content.endswith('# TODO'),
            'INCOMPLETE' in content.upper(),
            '# FRAGMENT' in content.upper(),
        ]
        
        return any(incomplete_signs)
    except:
        return False

def is_duplicate_logic(full_path, filename):
    """Identify files with duplicate logic"""
    duplicate_patterns = [
        'analyze_telemetry.py',  # Analysis output, not core logic
        lambda f: f.endswith('_old.py') or f.endswith('_backup.py'),
        lambda f: f.endswith('_v1.py') or f.endswith('_v2.py'),
        lambda f: 'duplicate' in f.lower(),
        lambda f: 'copy' in f.lower() and f.endswith('.py'),
    ]
    
    for pattern in duplicate_patterns:
        if callable(pattern):
            if pattern(filename):
                return True
        elif pattern == filename:
            return True
    
    return False

def is_analysis_output(full_path, filename):
    """Identify generated analysis/output files"""
    # Check if file is in telemetry output directory
    if 'telemetry' in full_path and not filename.endswith('.py'):
        return True
    
    analysis_patterns = [
        lambda f: f.startswith('analysis_') and f.endswith('.py'),
        lambda f: f.startswith('report_'),
        lambda f: f.endswith('.log') or f.endswith('.csv'),
        lambda f: f.endswith('_output.json') or f.endswith('_results.json'),
        lambda f: f in ['cleanup_log.txt', 'invocation_log.txt'],
    ]
    
    return any(pattern(filename) for pattern in analysis_patterns if callable(pattern))

def is_temp_file(full_path, filename):
    """Identify temporary files"""
    temp_patterns = [
        lambda f: f.startswith('.') and f != '.gitignore',  # Hidden files except gitignore
        lambda f: f.endswith('.tmp') or f.endswith('.temp'),
        lambda f: f.endswith('~') or f.startswith('~'),
        lambda f: f == '__pycache__',
        lambda f: f.endswith('.pyc') or f.endswith('.pyo'),
        lambda f: f == 'Thumbs.db' or f == '.DS_Store',
    ]
    
    return any(pattern(filename) for pattern in temp_patterns if callable(pattern))

def move_files_to_trash(analysis):
    """Move identified files to appropriate trash subdirectories"""
    categories = {
        "fragmented_files": analysis["fragmented_files"],
        "duplicate_logic": analysis["duplicate_logic"], 
        "analysis_outputs": analysis["analysis_outputs"],
        "temp_files": analysis["temp_files"]
    }
    
    moved_count = 0
    failed_count = 0
    
    for category, files in categories.items():
        if not files:
            continue
            
        logger.info(f"\n📁 Moving {len(files)} files to {category}:")
        
        for rel_path in files:
            try:
                source_path = os.path.join(SOURCE_DIR, rel_path)
                
                # Preserve directory structure in trash
                trash_subdir = os.path.join(TRASH_DIR, category)
                dest_path = os.path.join(trash_subdir, rel_path)
                
                # Create destination directory if needed
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Move the file
                shutil.move(source_path, dest_path)
                logger.info(f"   ✅ {rel_path}")
                moved_count += 1
                
            except Exception as e:
                logger.error(f"   ❌ Failed to move {rel_path}: {e}")
                failed_count += 1
    
    return moved_count, failed_count

def create_cleanup_report(analysis, moved_count, failed_count):
    """Create a detailed cleanup report"""
    report = {
        "cleanup_timestamp": datetime.now().isoformat(),
        "source_directory": SOURCE_DIR,
        "trash_directory": TRASH_DIR,
        "summary": {
            "total_files_analyzed": analysis["total_files"],
            "files_identified_for_cleanup": analysis["files_to_move"],
            "files_successfully_moved": moved_count,
            "files_failed_to_move": failed_count,
            "files_preserved": analysis["total_files"] - moved_count
        },
        "categories": {
            "fragmented_files": len(analysis["fragmented_files"]),
            "duplicate_logic": len(analysis["duplicate_logic"]),
            "analysis_outputs": len(analysis["analysis_outputs"]),
            "temp_files": len(analysis["temp_files"])
        }
    }
    
    # Save report
    report_path = os.path.join(TRASH_DIR, "cleanup_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return report

def print_summary(report):
    """Print a summary of the cleanup operation"""
    print("\n" + "="*60)
    print("🧹 SCAFAD LAYER 0 CLEANUP SUMMARY")
    print("="*60)
    
    summary = report["summary"]
    print(f"📊 Files analyzed: {summary['total_files_analyzed']}")
    print(f"🎯 Files identified for cleanup: {summary['files_identified_for_cleanup']}")
    print(f"✅ Files successfully moved: {summary['files_successfully_moved']}")
    print(f"❌ Files failed to move: {summary['files_failed_to_move']}")
    print(f"💾 Files preserved: {summary['files_preserved']}")
    
    print(f"\n📁 Breakdown by category:")
    for category, count in report["categories"].items():
        if count > 0:
            print(f"   • {category.replace('_', ' ').title()}: {count} files")
    
    print(f"\n📍 Locations:")
    print(f"   • Cleaned directory: {SOURCE_DIR}")
    print(f"   • Trash directory: {TRASH_DIR}")
    
    print(f"\n🎓 Next steps for dissertation:")
    print(f"   1. Review preserved files in source directory")
    print(f"   2. Reorganize remaining files using recommended structure")
    print(f"   3. Files can be restored from trash if needed")

def main():
    """Main cleanup function"""
    print("🧹 SCAFAD Layer 0 Cleanup Script")
    print("="*40)
    print(f"Source: {SOURCE_DIR}")
    print(f"Trash:  {TRASH_DIR}")
    print()
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return
    
    # Confirm operation
    response = input("This will move files to trash. Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    # Step 1: Create trash structure
    print("🗂️ Creating trash directory structure...")
    create_trash_structure()
    
    # Step 2: Backup current state
    print("💾 Creating backup of current state...")
    if not backup_current_state():
        print("❌ Backup failed. Cleanup cancelled for safety.")
        return
    
    # Step 3: Analyze files
    print("🔍 Analyzing files...")
    analysis = analyze_files()
    
    if analysis["files_to_move"] == 0:
        print("✅ No files identified for cleanup. Directory is already clean!")
        return
    
    # Step 4: Show what will be moved
    print(f"\n📋 Files to be moved ({analysis['files_to_move']} total):")
    categories = ["fragmented_files", "duplicate_logic", "analysis_outputs", "temp_files"]
    for category in categories:
        files = analysis[category]
        if files:
            print(f"\n📁 {category.replace('_', ' ').title()} ({len(files)} files):")
            for file in files[:5]:  # Show first 5
                print(f"   • {file}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
    
    # Final confirmation
    response = input(f"\nMove {analysis['files_to_move']} files to trash? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    # Step 5: Move files
    print("🚚 Moving files to trash...")
    moved_count, failed_count = move_files_to_trash(analysis)
    
    # Step 6: Create report
    print("📝 Creating cleanup report...")
    report = create_cleanup_report(analysis, moved_count, failed_count)
    
    # Step 7: Print summary
    print_summary(report)
    
    if failed_count > 0:
        print(f"\n⚠️ {failed_count} files could not be moved. Check cleanup_log.txt for details.")
    else:
        print(f"\n🎉 Cleanup completed successfully!")

if __name__ == "__main__":
    main()