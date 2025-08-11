#!/usr/bin/env python3
"""
SCAFAD Layer 0: Simple Validation Utilities
==========================================

Basic validation functions to ensure imports work correctly across all Layer 0 components.
This addresses the simplest import and validation issues for 100% submission.
"""

import sys
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def validate_basic_imports() -> Dict[str, bool]:
    """Validate that all basic SCAFAD imports work correctly"""
    
    validation_results = {
        'app_config': False,
        'app_telemetry': False,
        'app_formal': False,
        'core_graph_robustness': False,
        'core_crypto_validator': False
    }
    
    # Test app_config import
    try:
        from app_config import Layer0Config, FormalVerificationConfig
        config = Layer0Config()
        formal_config = FormalVerificationConfig()
        validation_results['app_config'] = True
        logger.debug("✅ app_config imports working")
    except ImportError as e:
        logger.warning(f"❌ app_config import failed: {e}")
    
    # Test app_telemetry import  
    try:
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase
        validation_results['app_telemetry'] = True
        logger.debug("✅ app_telemetry imports working")
    except ImportError as e:
        logger.warning(f"❌ app_telemetry import failed: {e}")
    
    # Test app_formal import
    try:
        from app_formal import FormalVerificationEngine
        validation_results['app_formal'] = True  
        logger.debug("✅ app_formal imports working")
    except ImportError as e:
        logger.warning(f"❌ app_formal import failed: {e}")
    
    # Test core graph robustness import
    try:
        from core.graph_robustness_analyzer import GraphRobustnessAnalyzer
        validation_results['core_graph_robustness'] = True
        logger.debug("✅ core.graph_robustness_analyzer imports working")
    except ImportError as e:
        logger.warning(f"❌ core.graph_robustness_analyzer import failed: {e}")
    
    # Test core crypto validator import
    try:
        from core.telemetry_crypto_validator import CryptographicTelemetryPipeline
        validation_results['core_crypto_validator'] = True
        logger.debug("✅ core.telemetry_crypto_validator imports working")
    except ImportError as e:
        logger.warning(f"❌ core.telemetry_crypto_validator import failed: {e}")
    
    return validation_results

def validate_component_creation() -> Dict[str, bool]:
    """Validate that all core components can be created without errors"""
    
    creation_results = {
        'layer0_config': False,
        'formal_verification_config': False,
        'telemetry_record': False,
        'graph_robustness_analyzer': False,
        'crypto_pipeline': False
    }
    
    # Test Layer0Config creation
    try:
        from app_config import Layer0Config
        config = Layer0Config()
        creation_results['layer0_config'] = True
        logger.debug("✅ Layer0Config created successfully")
    except Exception as e:
        logger.warning(f"❌ Layer0Config creation failed: {e}")
    
    # Test FormalVerificationConfig creation
    try:
        from app_config import FormalVerificationConfig
        formal_config = FormalVerificationConfig()
        creation_results['formal_verification_config'] = True
        logger.debug("✅ FormalVerificationConfig created successfully")
    except Exception as e:
        logger.warning(f"❌ FormalVerificationConfig creation failed: {e}")
    
    # Test TelemetryRecord creation
    try:
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase
        import time
        
        record = TelemetryRecord(
            event_id="test_001",
            timestamp=time.time(),
            function_id="test_func",
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=AnomalyType.BENIGN,
            duration=1.0,
            memory_spike_kb=1024,
            cpu_utilization=50.0,
            network_io_bytes=1000
        )
        creation_results['telemetry_record'] = True
        logger.debug("✅ TelemetryRecord created successfully")
    except Exception as e:
        logger.warning(f"❌ TelemetryRecord creation failed: {e}")
    
    # Test GraphRobustnessAnalyzer creation
    try:
        from core.graph_robustness_analyzer import GraphRobustnessAnalyzer
        analyzer = GraphRobustnessAnalyzer()
        creation_results['graph_robustness_analyzer'] = True
        logger.debug("✅ GraphRobustnessAnalyzer created successfully")
    except Exception as e:
        logger.warning(f"❌ GraphRobustnessAnalyzer creation failed: {e}")
    
    # Test CryptographicTelemetryPipeline creation
    try:
        from core.telemetry_crypto_validator import CryptographicTelemetryPipeline
        pipeline = CryptographicTelemetryPipeline()
        creation_results['crypto_pipeline'] = True
        logger.debug("✅ CryptographicTelemetryPipeline created successfully")
    except Exception as e:
        logger.warning(f"❌ CryptographicTelemetryPipeline creation failed: {e}")
    
    return creation_results

def run_simple_validation() -> Dict[str, Any]:
    """Run simple validation of all Layer 0 components"""
    
    logger.info("🔍 Running simple Layer 0 component validation")
    
    import_results = validate_basic_imports()
    creation_results = validate_component_creation()
    
    total_imports = len(import_results)
    successful_imports = sum(import_results.values())
    
    total_creations = len(creation_results)  
    successful_creations = sum(creation_results.values())
    
    overall_success = (successful_imports == total_imports and 
                      successful_creations == total_creations)
    
    summary = {
        'overall_success': overall_success,
        'import_validation': {
            'total_tests': total_imports,
            'successful': successful_imports,
            'success_rate': successful_imports / total_imports,
            'details': import_results
        },
        'creation_validation': {
            'total_tests': total_creations,
            'successful': successful_creations,
            'success_rate': successful_creations / total_creations,
            'details': creation_results
        }
    }
    
    if overall_success:
        logger.info("✅ All Layer 0 component validations passed!")
    else:
        failed_imports = [k for k, v in import_results.items() if not v]
        failed_creations = [k for k, v in creation_results.items() if not v]
        
        if failed_imports:
            logger.warning(f"❌ Failed imports: {failed_imports}")
        if failed_creations:
            logger.warning(f"❌ Failed creations: {failed_creations}")
    
    return summary

if __name__ == "__main__":
    """Run simple validation if called directly"""
    
    # Configure logging for standalone run
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔍 SCAFAD Layer 0 - Simple Component Validation")
    print("=" * 50)
    
    results = run_simple_validation()
    
    print(f"\nImport Validation: {results['import_validation']['successful']}/{results['import_validation']['total_tests']} passed")
    print(f"Creation Validation: {results['creation_validation']['successful']}/{results['creation_validation']['total_tests']} passed")
    
    if results['overall_success']:
        print("\n🎉 SIMPLE VALIDATION: PASSED")
        print("✅ All core components import and create successfully")
        sys.exit(0)
    else:
        print("\n❌ SIMPLE VALIDATION: FAILED")  
        print("⚠️ Some components have import or creation issues")
        
        # Show specific failures
        import_failures = [k for k, v in results['import_validation']['details'].items() if not v]
        creation_failures = [k for k, v in results['creation_validation']['details'].items() if not v]
        
        if import_failures:
            print(f"Failed Imports: {', '.join(import_failures)}")
        if creation_failures:
            print(f"Failed Creations: {', '.join(creation_failures)}")
        
        sys.exit(1)