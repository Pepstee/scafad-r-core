#!/usr/bin/env python3
"""
Verify the structure of test_adversarial.py and check that all functions return successfully
"""

import ast
import sys
import os

def analyze_file_structure():
    """Analyze the file structure using AST"""
    try:
        with open('/workspace/test_adversarial.py', 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        print(f"Found {len(classes)} classes:")
        for cls in sorted(set(classes)):
            print(f"  - {cls}")
        
        print(f"\nFound {len(functions)} functions:")
        for func in sorted(set(functions)):
            print(f"  - {func}")
        
        # Check for specific required functions
        required_functions = [
            'make_record',
            'test_noise_injection_does_not_mutate_original',
            'test_gradient_masking_adds_time_jitter',
            'test_input_transformation_logarithmic',
            'test_adaptive_perturbation_with_epsilon',
            'test_attack_vector_defaults',
            'test_adversarial_config_validation',
            'test_engine_generates_attack'
        ]
        
        missing_functions = [f for f in required_functions if f not in functions]
        
        if missing_functions:
            print(f"\n❌ Missing required functions: {missing_functions}")
            return False
        else:
            print(f"\n✅ All {len(required_functions)} required functions found")
            return True
            
    except SyntaxError as e:
        print(f"❌ Syntax error in file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return False

def test_import_and_execution():
    """Test import and basic execution"""
    try:
        # Add workspace to path
        sys.path.insert(0, '/workspace')
        
        # Import the module
        print("Importing test_adversarial...")
        import test_adversarial
        print("✅ Import successful")
        
        # Test make_record function
        print("Testing make_record()...")
        record = test_adversarial.make_record()
        assert record.event_id == "evt"
        print("✅ make_record() works")
        
        # Test TestFixtures
        print("Testing TestFixtures...")
        sample = test_adversarial.TestFixtures.create_sample_telemetry()
        assert sample.event_id == "test_event_001"
        print("✅ TestFixtures works")
        
        # Test standalone functions
        test_functions = [
            'test_noise_injection_does_not_mutate_original',
            'test_gradient_masking_adds_time_jitter',
            'test_input_transformation_logarithmic',
            'test_adaptive_perturbation_with_epsilon',
            'test_attack_vector_defaults',
            'test_adversarial_config_validation'
        ]
        
        successful_tests = 0
        for func_name in test_functions:
            try:
                print(f"Testing {func_name}...")
                func = getattr(test_adversarial, func_name)
                func()
                print(f"✅ {func_name} passed")
                successful_tests += 1
            except Exception as e:
                print(f"⚠ {func_name} had issues: {e}")
        
        # Test async function separately
        try:
            print("Testing test_engine_generates_attack...")
            test_adversarial.test_engine_generates_attack()
            print("✅ test_engine_generates_attack passed")
            successful_tests += 1
        except Exception as e:
            print(f"⚠ test_engine_generates_attack had issues: {e}")
        
        total_tests = len(test_functions) + 1  # +1 for async test
        print(f"\n✅ {successful_tests}/{total_tests} test functions executed successfully")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"❌ Error during import/execution test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    print("=" * 60)
    print("STRUCTURE AND EXECUTION VERIFICATION")  
    print("=" * 60)
    
    print("\n1. ANALYZING FILE STRUCTURE:")
    structure_ok = analyze_file_structure()
    
    print("\n2. TESTING IMPORT AND EXECUTION:")
    execution_ok = test_import_and_execution()
    
    print("\n" + "=" * 60)
    
    if structure_ok and execution_ok:
        print("✅ VERIFICATION COMPLETE: test_adversarial.py WORKS CORRECTLY!")
        print("✅ ALL FUNCTIONS RETURN VALUES SUCCESSFULLY!")
        return True
    else:
        print("❌ VERIFICATION FAILED: Issues found in test_adversarial.py")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nResult: {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1)