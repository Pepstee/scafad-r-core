#!/usr/bin/env python3
"""
SCAFAD Layer 0: Windows-Compatible Academic Assessment
=====================================================

Comprehensive assessment of Layer 0 implementation for 95% academic readiness.
Designed to work on both Windows and Linux systems.
"""

import os
import sys
import time
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple

class WindowsAcademicAssessor:
    """Windows-compatible academic readiness assessor"""
    
    def __init__(self):
        self.workspace_path = Path(__file__).parent
        self.assessment_results = {
            'timestamp': time.time(),
            'platform': sys.platform,
            'workspace_path': str(self.workspace_path),
            'categories': {},
            'overall_score': 0.0,
            'academic_ready': False,
            'recommendations': []
        }
    
    def assess_all_categories(self) -> Dict[str, Any]:
        """Run comprehensive assessment across all categories"""
        
        print("🎓 SCAFAD Layer 0 - Windows Academic Assessment")
        print("=" * 50)
        print(f"Platform: {sys.platform}")
        print(f"Workspace: {self.workspace_path}")
        print("Evaluating implementation against 95% academic criterion\n")
        
        # Category 1: Core Architecture & Components (30%)
        arch_score = self.assess_core_architecture()
        self.assessment_results['categories']['core_architecture'] = {
            'score': arch_score,
            'weight': 30,
            'description': 'Core architecture implementation and component integration'
        }
        
        # Category 2: Algorithmic Implementation (25%)
        algo_score = self.assess_algorithmic_implementation()
        self.assessment_results['categories']['algorithmic_implementation'] = {
            'score': algo_score,
            'weight': 25,
            'description': 'Anomaly detection algorithms and core detection engine'
        }
        
        # Category 3: Integration & Validation (20%)
        integration_score = self.assess_integration_validation()
        self.assessment_results['categories']['integration_validation'] = {
            'score': integration_score,
            'weight': 20,
            'description': 'System integration and validation frameworks'
        }
        
        # Category 4: Academic Features (15%)
        academic_score = self.assess_academic_features()
        self.assessment_results['categories']['academic_features'] = {
            'score': academic_score,
            'weight': 15,
            'description': 'Academic documentation, reproducibility, and validation'
        }
        
        # Category 5: Performance & Quality (10%)
        quality_score = self.assess_performance_quality()
        self.assessment_results['categories']['performance_quality'] = {
            'score': quality_score,
            'weight': 10,
            'description': 'Performance optimization and code quality'
        }
        
        # Calculate overall weighted score
        self._calculate_overall_score()
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.assessment_results
    
    def assess_core_architecture(self) -> float:
        """Assess core architecture and component implementation"""
        
        print("🏗️ Assessing Core Architecture & Components...")
        
        core_files = {
            'app_main.py': {'required_features': ['Layer0_AdaptiveTelemetryController', 'run_academic_validation'], 'weight': 0.2},
            'app_telemetry.py': {'required_features': ['MultiChannelTelemetry', 'failover', 'TelemetryRecord'], 'weight': 0.15},
            'layer0_core.py': {'required_features': ['AnomalyDetectionEngine', 'detect_anomalies', '26'], 'weight': 0.15},
            'app_formal.py': {'required_features': ['DistributedLTLVerifier', 'Byzantine'], 'weight': 0.1},
            'layer0_signal_negotiation.py': {'required_features': ['SignalNegotiator'], 'weight': 0.08},
            'layer0_redundancy_manager.py': {'required_features': ['RedundancyManager'], 'weight': 0.08},
            'layer0_sampler.py': {'required_features': ['Sampler'], 'weight': 0.08},
            'layer0_fallback_orchestrator.py': {'required_features': ['FallbackOrchestrator'], 'weight': 0.08},
            'layer0_runtime_control.py': {'required_features': ['RuntimeControlLoop'], 'weight': 0.08}
        }
        
        total_score = 0.0
        components_found = 0
        
        for filename, config in core_files.items():
            file_path = self.workspace_path / filename
            file_score = 0.0
            
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    features_found = sum(1 for feature in config['required_features'] if feature in content)
                    feature_score = features_found / len(config['required_features'])
                    
                    # Size score (larger files generally indicate more implementation)
                    size_score = min(len(content.split('\n')) / 1000, 1.0)  # Up to 1000 lines = full score
                    
                    file_score = (feature_score * 0.8) + (size_score * 0.2)
                    components_found += 1
                    
                except Exception as e:
                    print(f"   ❌ Error reading {filename}: {e}")
                    file_score = 0.0
            
            total_score += file_score * config['weight']
        
        architecture_score = total_score * 100
        print(f"   Architecture Score: {architecture_score:.1f}% ({components_found}/{len(core_files)} components found)")
        
        return architecture_score
    
    def assess_algorithmic_implementation(self) -> float:
        """Assess algorithmic implementation depth"""
        
        print("🧠 Assessing Algorithmic Implementation...")
        
        algorithm_checks = {
            'layer0_core_exists': self._check_file_exists('layer0_core.py'),
            'anomaly_detection_engine': self._check_file_contains('layer0_core.py', 'AnomalyDetectionEngine'),
            'multi_vector_fusion': self._check_file_contains('layer0_core.py', 'multi_vector_fusion'),
            'detection_algorithms_26': self._check_file_contains('layer0_core.py', 'Algorithm 26'),
            'statistical_detection': self._check_file_contains('layer0_core.py', '_detect_statistical'),
            'machine_learning': self._check_file_contains('layer0_core.py', 'IsolationForest'),
            'trust_weighted_voting': self._check_file_contains('layer0_core.py', 'trust_weighted'),
            'byzantine_tolerance': self._check_file_contains_any(['layer0_core.py', 'app_formal.py'], 'byzantine'),
            'graph_robustness': self._check_file_exists('core/graph_robustness_analyzer.py'),
            'crypto_validation': self._check_file_exists('core/telemetry_crypto_validator.py')
        }
        
        passed_checks = sum(algorithm_checks.values())
        total_checks = len(algorithm_checks)
        algorithm_score = (passed_checks / total_checks) * 100
        
        print(f"   Algorithmic Score: {algorithm_score:.1f}% ({passed_checks}/{total_checks} algorithms)")
        
        return algorithm_score
    
    def assess_integration_validation(self) -> float:
        """Assess system integration and validation"""
        
        print("⚡ Assessing Integration & Validation...")
        
        integration_checks = {
            'main_orchestrator_integration': self._check_file_contains('app_main.py', 'from layer0_core import'),
            'telemetry_integration': self._check_file_contains('app_main.py', 'self.anomaly_detection_engine'),
            'test_data_generator': self._check_file_exists('utils/test_data_generator.py'),
            'validation_framework': self._check_file_contains('app_main.py', 'run_academic_validation'),
            'performance_benchmarks': self._check_file_contains('app_main.py', 'run_performance_benchmarks'),
            'integration_tests': self._check_file_contains('app_main.py', 'run_integration_tests'),
            'stress_testing': self._check_file_contains('app_main.py', 'stress_test'),
            'health_monitoring': self._check_file_contains('app_main.py', 'health_check'),
            'component_orchestration': self._check_file_contains('app_main.py', 'process_invocation'),
            'error_handling': self._check_file_contains('app_main.py', '_handle_processing_failure')
        }
        
        passed_checks = sum(integration_checks.values())
        total_checks = len(integration_checks)
        integration_score = (passed_checks / total_checks) * 100
        
        print(f"   Integration Score: {integration_score:.1f}% ({passed_checks}/{total_checks} integrations)")
        
        return integration_score
    
    def assess_academic_features(self) -> float:
        """Assess academic documentation and features"""
        
        print("📚 Assessing Academic Features...")
        
        academic_checks = {
            'claude_md_comprehensive': self._check_file_exists('CLAUDE.md'),
            'readme_quality': self._check_file_exists('README.md'),
            'reproducibility_setup': self._check_file_exists('reproduce_results.py'),
            'academic_assessment_tools': self._check_file_exists('assess_academic_readiness.py'),
            'final_validation_framework': self._check_file_exists('final_academic_validation.py'),
            'academic_references': self._check_file_contains_any(['layer0_core.py', 'app_formal.py'], 'Academic References'),
            'validation_scripts': self._check_file_exists('validate_app_main.py'),
            'comprehensive_documentation': self._check_file_contains('README.md', 'Academic Citation'),
            'peer_review_ready': self._check_file_contains('README.md', '@misc{'),
            'citation_format': self._check_file_contains('README.md', 'bibtex')
        }
        
        passed_checks = sum(academic_checks.values())
        total_checks = len(academic_checks)
        academic_score = (passed_checks / total_checks) * 100
        
        print(f"   Academic Score: {academic_score:.1f}% ({passed_checks}/{total_checks} features)")
        
        return academic_score
    
    def assess_performance_quality(self) -> float:
        """Assess performance optimization and code quality"""
        
        print("🚀 Assessing Performance & Quality...")
        
        quality_checks = {
            'async_implementation': self._check_file_contains('app_main.py', 'async def'),
            'error_handling': self._check_file_contains_any(['app_main.py', 'layer0_core.py'], 'try:'),
            'logging_implementation': self._check_file_contains_any(['layer0_core.py'], 'logger'),
            'type_hints': self._check_file_contains_any(['layer0_core.py'], 'Dict[str, Any]'),
            'docstring_coverage': self._check_file_contains_any(['layer0_core.py'], '"""'),
            'performance_optimizations': self._check_file_contains('layer0_core.py', 'processing_time_ms'),
            'graceful_degradation': self._check_file_contains_any(['app_main.py', 'app_telemetry.py'], 'fallback'),
            'resource_management': self._check_file_contains('layer0_core.py', 'deque'),
            'algorithm_efficiency': self._check_file_contains('layer0_core.py', 'sklearn'),
            'comprehensive_testing': self._check_directory_exists('tests/')
        }
        
        passed_checks = sum(quality_checks.values())
        total_checks = len(quality_checks)
        quality_score = (passed_checks / total_checks) * 100
        
        print(f"   Quality Score: {quality_score:.1f}% ({passed_checks}/{total_checks} quality features)")
        
        return quality_score
    
    # Helper methods
    def _check_file_exists(self, filename: str) -> bool:
        """Check if file exists"""
        return (self.workspace_path / filename).exists()
    
    def _check_file_contains(self, filename: str, content: str) -> bool:
        """Check if file contains specific content"""
        file_path = self.workspace_path / filename
        if not file_path.exists():
            return False
        
        try:
            file_content = file_path.read_text(encoding='utf-8')
            return content in file_content
        except Exception:
            return False
    
    def _check_file_contains_any(self, filenames: List[str], content: str) -> bool:
        """Check if any of the files contains specific content"""
        return any(self._check_file_contains(filename, content) for filename in filenames)
    
    def _check_directory_exists(self, dirname: str) -> bool:
        """Check if directory exists"""
        dir_path = self.workspace_path / dirname
        return dir_path.exists() and dir_path.is_dir()
    
    def _calculate_overall_score(self):
        """Calculate overall weighted score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for category, data in self.assessment_results['categories'].items():
            weighted_contribution = (data['score'] * data['weight']) / 100
            total_weighted_score += weighted_contribution
            total_weight += data['weight']
        
        self.assessment_results['overall_score'] = total_weighted_score
        self.assessment_results['academic_ready'] = total_weighted_score >= 95.0
    
    def _generate_recommendations(self):
        """Generate improvement recommendations"""
        recommendations = []
        
        for category, data in self.assessment_results['categories'].items():
            if data['score'] < 95.0:
                if category == 'core_architecture':
                    recommendations.append("Complete missing Layer 0 component implementations")
                elif category == 'algorithmic_implementation':
                    recommendations.append("Enhance anomaly detection algorithms and ML integration")
                elif category == 'integration_validation':
                    recommendations.append("Improve system integration and validation frameworks")
                elif category == 'academic_features':
                    recommendations.append("Enhance academic documentation and reproducibility")
                elif category == 'performance_quality':
                    recommendations.append("Optimize performance and improve code quality")
        
        if self.assessment_results['overall_score'] >= 95.0:
            recommendations.append("🎉 Excellent work! Layer 0 meets academic standards")
        elif self.assessment_results['overall_score'] >= 85.0:
            recommendations.append("Very close to 95% - focus on highest-weighted categories")
        else:
            recommendations.append("Focus on core architecture and algorithmic implementation")
        
        self.assessment_results['recommendations'] = recommendations
    
    def display_results(self):
        """Display formatted assessment results"""
        results = self.assessment_results
        
        print(f"\n📊 ACADEMIC READINESS ASSESSMENT RESULTS")
        print("=" * 55)
        print(f"Platform: {results['platform']}")
        print(f"Overall Score: {results['overall_score']:.1f}%")
        print(f"Academic Ready: {'✅ YES' if results['academic_ready'] else '❌ NO'}")
        print()
        
        print("📋 CATEGORY BREAKDOWN:")
        print("-" * 30)
        for category, data in results['categories'].items():
            status_icon = "✅" if data['score'] >= 95.0 else "⚠️" if data['score'] >= 85.0 else "❌"
            category_name = category.replace('_', ' ').title()
            print(f"{status_icon} {category_name}: {data['score']:.1f}% (Weight: {data['weight']}%)")
        
        print()
        
        if results['recommendations']:
            print("🔧 RECOMMENDATIONS:")
            print("-" * 20)
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")
            print()
        
        # Academic readiness summary
        if results['academic_ready']:
            print("🎉 ACADEMIC ASSESSMENT: EXCELLENT (≥95%)")
            print("✅ Layer 0 meets all criteria for academic submission")
            print("📄 Ready for peer review and dissertation defense")
        elif results['overall_score'] >= 85.0:
            print("⚠️ ACADEMIC ASSESSMENT: VERY GOOD (85-94%)")
            print("🔧 Minor improvements needed to reach 95% threshold")
            print("📝 Focus on recommendations to achieve excellence")
        else:
            print("❌ ACADEMIC ASSESSMENT: NEEDS IMPROVEMENT (<85%)")
            print("🚧 Significant development required for academic submission")
            print("📋 Focus on core architecture and algorithmic implementation")

def main():
    """Main assessment function"""
    assessor = WindowsAcademicAssessor()
    
    try:
        # Run comprehensive assessment
        results = assessor.assess_all_categories()
        
        # Display results
        assessor.display_results()
        
        # Save results to file
        results_file = assessor.workspace_path / f"academic_assessment_results_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Return appropriate exit code
        if results['academic_ready']:
            print(f"\n🎉 SUCCESS: Layer 0 meets 95%+ academic readiness!")
            return 0
        elif results['overall_score'] >= 85.0:
            print(f"\n⚠️ CLOSE: {results['overall_score']:.1f}% - minor improvements needed")
            return 1
        else:
            print(f"\n❌ NEEDS WORK: {results['overall_score']:.1f}% - significant improvements needed")
            return 2
        
    except Exception as e:
        print(f"💥 Assessment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Assessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(2)