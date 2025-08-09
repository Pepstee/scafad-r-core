"""
SCAFAD Layer 0: Comprehensive Adversarial Test Suite
==================================================

Comprehensive testing framework for the adversarial simulation engine.
Includes unit tests, integration tests, performance tests, and academic validation.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: End-to-end workflow testing  
- Performance Tests: Scalability and efficiency testing
- Academic Tests: Research validation and statistical testing
- Security Tests: Attack realism and effectiveness validation
"""

import pytest
import asyncio
import time
import numpy as np
import random
import logging
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import json
from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
import random



# Import the modules being tested
from app_adversarial import (
    AdversarialAnomalyEngine,
    AttackType,
    AttackVector,
    AttackResult,
    ServerlessAnomalyGAN,
    AdaptiveAttacker,
    EvasionTechniques,
    PoisoningAttackGenerator,
    EconomicAttackSimulator,
    AdversarialTestSuite,
    AdversarialMetricsCollector,
    AdversarialTrainingManager,
    QueryFreeAttackEngine,
    TransferAttackEngine,
    AdversarialValidationFramework,
    MultiStepCampaignOrchestrator,
    AdversarialRobustnessAnalyzer
)

from app_config import AdversarialConfig, AdversarialMode
from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestFixtures:
    """Test utility class providing common test data and configurations"""
    
    @staticmethod
    def create_sample_telemetry() -> TelemetryRecord:
        """Create a sample telemetry record for testing"""
        return TelemetryRecord(
            event_id="test_event_001",
            timestamp=time.time(),
            function_id="test_function",
            execution_phase=ExecutionPhase.INVOKE,  # Fixed: MAIN -> INVOKE
            anomaly_type=AnomalyType.BENIGN,
            duration=1.5,
            memory_spike_kb=128000,
            cpu_utilization=45.2,
            network_io_bytes=2048,
            # Required parameters that were missing:
            fallback_mode=False,
            source=TelemetrySource.PRIMARY,
            concurrency_id="test_concurrency_001"
        )
    
    @staticmethod
    def create_anomalous_telemetry() -> TelemetryRecord:
        """Create an anomalous telemetry record for testing"""
        return TelemetryRecord(
            event_id="test_anomaly_001",
            timestamp=time.time(),
            function_id="suspicious_function",
            execution_phase=ExecutionPhase.INVOKE,  # Fixed: MAIN -> INVOKE
            anomaly_type=AnomalyType.CPU_BURST,
            duration=12.5,
            memory_spike_kb=512000,
            cpu_utilization=95.8,
            network_io_bytes=1048576,
            # Required parameters:
            fallback_mode=False,
            source=TelemetrySource.PRIMARY,
            concurrency_id="test_concurrency_002"
        )
    
    @staticmethod
    def create_test_config() -> AdversarialConfig:
        """Create a test configuration"""
        return AdversarialConfig(
            adversarial_mode=AdversarialMode.TEST,  # Fixed: TESTING -> TEST
            enable_evasion_techniques=True,
            enable_poisoning_attacks=True,
            max_evasion_budget=0.1,
            max_poisoning_rate=0.05,
            enable_gan_generation=False,  # Disable for testing
            gan_epochs=10,
            gan_batch_size=16
        )
    
    @staticmethod
    def create_test_dataset(size: int) -> List[TelemetryRecord]:
        """Create a test dataset with specified size"""
        dataset = []
        for i in range(size):
            record = TelemetryRecord(
                event_id=f"test_dataset_event_{i}",
                timestamp=time.time() - (size - i) * 60,  # Space out timestamps
                function_id=f"test_function_{i % 5}",
                execution_phase=ExecutionPhase.INVOKE,
                anomaly_type=AnomalyType.BENIGN if i % 3 != 0 else AnomalyType.CPU_BURST,
                duration=random.uniform(0.1, 5.0),
                memory_spike_kb=random.randint(64000, 256000),
                cpu_utilization=random.uniform(10.0, 90.0),
                network_io_bytes=random.randint(1000, 50000),
                fallback_mode=False,
                source=TelemetrySource.PRIMARY,
                concurrency_id=f"test_concurrency_{i}"
            )
            dataset.append(record)
        return dataset


"""
[tool:pytest]
asyncio_mode = auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    -ra
"""

# Fix 3: Bug fixes for app_adversarial.py

# Fix for _assess_operational_risk method - add missing vulnerability parameter:
def _assess_operational_risk(self, robustness_scores: Dict[str, Dict[str, float]], 
                           test_results: Dict[str, Any],
                           vulnerabilities: Dict[str, Any] = None) -> Dict[str, Any]:
    """Assess operational risk based on robustness analysis"""
    
    # Provide default vulnerabilities if not passed
    if vulnerabilities is None:
        vulnerabilities = {'critical_vulnerabilities': []}
    
    risk_assessment = {
        'attack_success_probability': {},
        'economic_impact_risk': 0.0,
        'operational_risk_score': 0.0,
        'risk_mitigation_priority': [],
        'business_impact_assessment': {}
    }
    
    # Calculate attack success probabilities
    total_economic_impact = 0.0
    max_economic_impact = 0.0
    
    for attack_type, scores in robustness_scores.items():
        # Attack success probability = 1 - overall robustness
        success_prob = 1.0 - scores.get('overall_robustness', 0.0)
        risk_assessment['attack_success_probability'][attack_type] = success_prob
        
        # Calculate economic impact risk
        if 'attack_results' in test_results and attack_type in test_results['attack_results']:
            economic_impact = test_results['attack_results'][attack_type].get('economic_impact', 0.0)
            total_economic_impact += economic_impact * success_prob
            max_economic_impact = max(max_economic_impact, economic_impact)
    
    risk_assessment['economic_impact_risk'] = total_economic_impact
    
    # Calculate operational risk score
    avg_success_prob = np.mean(list(risk_assessment['attack_success_probability'].values())) if risk_assessment['attack_success_probability'] else 0.0
    economic_factor = min(1.0, max_economic_impact / 1000.0)  # Normalize to $1000
    
    risk_assessment['operational_risk_score'] = (
        0.6 * avg_success_prob +
        0.4 * economic_factor
    )
    
    # Generate risk mitigation priorities
    attack_risks = [
        (attack_type, prob, test_results.get('attack_results', {}).get(attack_type, {}).get('economic_impact', 0.0))
        for attack_type, prob in risk_assessment['attack_success_probability'].items()
    ]
    
    # Sort by combined risk (probability * economic impact)
    attack_risks.sort(key=lambda x: x[1] * x[2], reverse=True)
    
    risk_assessment['risk_mitigation_priority'] = [
        {
            'attack_type': attack_type,
            'success_probability': prob,
            'economic_impact': economic_impact,
            'combined_risk': prob * economic_impact,
            'priority_level': 'high' if prob * economic_impact > 50 else 'medium' if prob * economic_impact > 10 else 'low'
        }
        for attack_type, prob, economic_impact in attack_risks[:10]  # Top 10 risks
    ]
    
    # Business impact assessment
    risk_assessment['business_impact_assessment'] = {
        'availability_risk': 'high' if avg_success_prob > 0.7 else 'medium' if avg_success_prob > 0.4 else 'low',
        'financial_risk': 'high' if total_economic_impact > 500 else 'medium' if total_economic_impact > 100 else 'low',
        'reputation_risk': 'high' if len(vulnerabilities.get('critical_vulnerabilities', [])) > 0 else 'medium',
        'compliance_risk': 'medium'  # Default assessment
    }
    
    return risk_assessment

# Fix for comprehensive_robustness_analysis method:
def comprehensive_robustness_analysis(self, test_results: Dict[str, Any], 
                                    baseline_performance: Dict[str, float]) -> Dict[str, Any]:
    """Perform comprehensive robustness analysis"""
    
    analysis = {
        'analysis_timestamp': time.time(),
        'robustness_scores': {},
        'vulnerability_assessment': {},
        'improvement_recommendations': [],
        'benchmark_comparison': {},
        'risk_assessment': {}
    }
    
    # Calculate robustness scores for each attack type
    analysis['robustness_scores'] = self._calculate_robustness_scores(
        test_results, baseline_performance
    )
    
    # Assess vulnerabilities
    analysis['vulnerability_assessment'] = self._assess_vulnerabilities(
        analysis['robustness_scores']
    )
    
    # Generate improvement recommendations
    analysis['improvement_recommendations'] = self._generate_improvement_recommendations(
        analysis['robustness_scores'], analysis['vulnerability_assessment']
    )
    
    # Compare against benchmarks
    analysis['benchmark_comparison'] = self._compare_to_benchmarks(
        analysis['robustness_scores']
    )
    
    # Assess operational risk - pass vulnerabilities parameter
    analysis['risk_assessment'] = self._assess_operational_risk(
        analysis['robustness_scores'], test_results, analysis['vulnerability_assessment']
    )
    
    # Store analysis history
    self.analysis_history.append(analysis)
    
    return analysis

# Fix for _generate_improvement_recommendations method:
def _generate_improvement_recommendations(self, robustness_scores: Dict[str, Dict[str, float]], 
                                        vulnerabilities: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate specific improvement recommendations"""
    
    recommendations = []
    
    # Critical vulnerability recommendations
    for vuln in vulnerabilities.get('critical_vulnerabilities', []):
        recommendations.append({
            'priority': 'critical',
            'type': 'immediate_action',
            'attack_type': vuln.get('attack_type', 'unknown'),
            'recommendation': f"CRITICAL: Immediate attention required for {vuln.get('attack_type', 'unknown')} attacks. "
                            f"Current robustness: {vuln.get('robustness_score', 0.0):.2f}. "
                            f"Implement emergency countermeasures and enhanced monitoring."
        })
    
    # Pattern-based recommendations
    patterns = vulnerabilities.get('vulnerability_patterns', {})
    
    if patterns.get('detection_weakness'):
        recommendations.append({
            'priority': 'high',
            'type': 'detection_improvement',
            'attack_type': 'multiple',
            'recommendation': "Systemic detection weakness identified. Consider implementing "
                            "ensemble detection methods, improving feature engineering, "
                            "and adding behavioral analysis components."
        })
    
    if patterns.get('performance_weakness'):
        recommendations.append({
            'priority': 'medium',
            'type': 'performance_optimization',
            'attack_type': 'multiple',
            'recommendation': "Performance degradation under attack detected. Implement "
                            "adversarial training, input preprocessing, and robustness "
                            "regularization techniques."
        })
    
    if patterns.get('economic_weakness'):
        recommendations.append({
            'priority': 'high',
            'type': 'economic_protection',
            'attack_type': 'multiple',
            'recommendation': "Economic vulnerability detected. Implement cost monitoring, "
                            "rate limiting, resource quotas, and billing anomaly detection."
        })
    
    # Attack-specific recommendations
    for attack_type, scores in robustness_scores.items():
        overall_robustness = scores.get('overall_robustness', 1.0)
        if overall_robustness < 0.6:
            if 'evasion' in attack_type or 'perturbation' in attack_type:
                recommendations.append({
                    'priority': 'medium',
                    'type': 'evasion_resistance',
                    'attack_type': attack_type,
                    'recommendation': f"Improve evasion resistance for {attack_type}. "
                                    f"Implement input validation, adversarial training, "
                                    f"and ensemble defense methods."
                })
            elif 'economic' in attack_type or 'mining' in attack_type or 'wallet' in attack_type:
                recommendations.append({
                    'priority': 'high',
                    'type': 'economic_defense',
                    'attack_type': attack_type,
                    'recommendation': f"Strengthen economic defenses against {attack_type}. "
                                    f"Implement real-time cost monitoring, usage anomaly "
                                    f"detection, and automatic resource limiting."
                })
    
    # General recommendations if no specific issues found
    if not recommendations:
        recommendations.append({
            'priority': 'low',
            'type': 'maintenance',
            'attack_type': 'all',
            'recommendation': "System shows good robustness. Continue regular testing, "
                            "monitor for emerging attack vectors, and maintain current "
                            "security measures."
        })
    
    return recommendations

# Fix for research report generation in AdversarialMetricsCollector:
def generate_research_report(self) -> Dict[str, Any]:
    """Generate comprehensive research report on adversarial testing"""
    
    if not self.metrics_history:
        return {"error": "No metrics data available"}
    
    report = {
        'summary': {
            'total_attacks': len(self.metrics_history),
            'unique_attack_types': len(set(m['attack_type'] for m in self.metrics_history)),
            'overall_success_rate': np.mean([m['attack_success'] for m in self.metrics_history]),
            'average_stealth': np.mean([m['stealth_score'] for m in self.metrics_history]),
            'total_economic_impact': sum(m['economic_impact'] for m in self.metrics_history)
        },
        'attack_type_analysis': {},
        'performance_baselines': self.performance_baselines,
        'temporal_trends': {}  # Add this missing field
    }
    
    # Analyze by attack type
    attack_types = set(m['attack_type'] for m in self.metrics_history)
    for attack_type in attack_types:
        type_metrics = [m for m in self.metrics_history if m['attack_type'] == attack_type]
        
        report['attack_type_analysis'][attack_type] = {
            'count': len(type_metrics),
            'success_rate': np.mean([m['attack_success'] for m in type_metrics]),
            'avg_stealth': np.mean([m['stealth_score'] for m in type_metrics]),
            'avg_perturbation': np.mean([m['perturbation_magnitude'] for m in type_metrics]),
            'economic_impact': sum(m['economic_impact'] for m in type_metrics),
            'detection_rate': np.mean([m['detection_triggered'] for m in type_metrics])
        }
    
    # Add temporal trend analysis
    if len(self.metrics_history) > 1:
        timestamps = [m['timestamp'] for m in self.metrics_history]
        timestamps.sort()
        
        # Calculate trends over time
        window_size = max(5, len(self.metrics_history) // 4)
        recent_metrics = self.metrics_history[-window_size:]
        early_metrics = self.metrics_history[:window_size]
        
        recent_success_rate = np.mean([m['attack_success'] for m in recent_metrics])
        early_success_rate = np.mean([m['attack_success'] for m in early_metrics])
        
        report['temporal_trends'] = {
            'success_rate_trend': 'increasing' if recent_success_rate > early_success_rate + 0.1 
                                 else 'decreasing' if recent_success_rate < early_success_rate - 0.1 
                                 else 'stable',
            'recent_success_rate': recent_success_rate,
            'early_success_rate': early_success_rate,
            'time_span_hours': (timestamps[-1] - timestamps[0]) / 3600 if len(timestamps) > 1 else 0
        }
    else:
        report['temporal_trends'] = {
            'success_rate_trend': 'insufficient_data',
            'recent_success_rate': 0.0,
            'early_success_rate': 0.0,
            'time_span_hours': 0.0
        }
    
    return report

# Fix for missing engine attribute in TestPerformanceAndScalability:
class TestPerformanceAndScalability:
    """Test performance and scalability aspects"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = TestFixtures.create_test_config()
        self.engine = AdversarialAnomalyEngine(self.config)  # Add this line
        self.large_dataset_size = 1000
        self.concurrent_requests = 50


class TestAdversarialConfig:
    """Test adversarial configuration management"""
    
    def test_config_creation(self):
        """Test basic configuration creation"""
        config = TestFixtures.create_test_config()
        
        assert config.adversarial_mode == AdversarialMode.TEST 
        assert config.enable_gan_generation is False
        assert config.gan_latent_dim == 100 
        assert 0.0 <= config.max_evasion_budget <= 1.0
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = AdversarialConfig(
            gan_latent_dim=5,  # Too small
            gan_learning_rate=0.1,  # Too high
            max_evasion_budget=1.5,  # Invalid range
            max_poisoning_rate=0.15  # Too high
        )
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any("latent dimension too small" in issue for issue in issues)
        assert any("learning rate" in issue for issue in issues)
        assert any("evasion budget" in issue for issue in issues)
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = AdversarialConfig()
        
        assert config.adversarial_mode == AdversarialMode.DISABLED
        assert config.gan_latent_dim == 100
        assert config.gan_feature_dim == 20
        assert config.max_evasion_budget == 0.1


class TestEvasionTechniques:
    """Test evasion technique implementations"""
    
    def setup_method(self):
        """Setup for each test"""
        self.sample_telemetry = TestFixtures.create_sample_telemetry()
    
    def test_noise_injection(self):
        """Test noise injection evasion"""
        original = self.sample_telemetry
        perturbed = EvasionTechniques.noise_injection(original, noise_level=0.1)
        
        # Check that values changed but remain valid
        assert perturbed.duration != original.duration
        assert perturbed.duration > 0
        assert 0 <= perturbed.cpu_utilization <= 100
        assert perturbed.memory_spike_kb >= 0
        assert perturbed.network_io_bytes >= 0
    
    def test_gradient_masking(self):
        """Test gradient masking evasion"""
        original = self.sample_telemetry
        perturbed = EvasionTechniques.gradient_masking(original, masking_strength=0.2)
        
        # Check that temporal properties changed
        assert perturbed.timestamp != original.timestamp
        assert perturbed.duration != original.duration
        assert perturbed.duration > 0
    
    def test_input_transformation(self):
        """Test input transformation evasion"""
        original = self.sample_telemetry
        
        # Test different transformation types
        log_transformed = EvasionTechniques.input_transformation(original, "logarithmic")
        exp_transformed = EvasionTechniques.input_transformation(original, "exponential")
        quant_transformed = EvasionTechniques.input_transformation(original, "quantization")
        
        # Check transformations applied
        assert log_transformed.duration != original.duration
        assert exp_transformed.cpu_utilization != original.cpu_utilization
        assert quant_transformed.memory_spike_kb != original.memory_spike_kb
    
    def test_adaptive_perturbation(self):
        """Test adaptive perturbation evasion"""
        original = self.sample_telemetry
        perturbed = EvasionTechniques.adaptive_perturbation(original, epsilon=0.1)
        
        # Check that all numerical fields were perturbed
        assert perturbed.duration != original.duration
        assert perturbed.memory_spike_kb != original.memory_spike_kb
        assert perturbed.cpu_utilization != original.cpu_utilization
        assert perturbed.network_io_bytes != original.network_io_bytes
        
        # Check valid ranges maintained
        assert perturbed.duration > 0
        assert 0 <= perturbed.cpu_utilization <= 100
        assert perturbed.memory_spike_kb >= 0


class TestPoisoningAttackGenerator:
    """Test poisoning attack generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.generator = PoisoningAttackGenerator(max_poison_rate=0.05)
        self.test_data = TestFixtures.create_test_dataset(20)
    
    def test_label_flip_attack(self):
        """Test label flipping poisoning attack"""
        poison_rate = 0.03
        poisoned_data = self.generator.generate_label_flip_attack(self.test_data, poison_rate)
        
        # Check poison rate
        poisoned_count = sum(1 for record in poisoned_data 
                           if record.custom_fields.get('poisoned', False))
        actual_poison_rate = poisoned_count / len(poisoned_data)
        
        assert 0 <= actual_poison_rate <= poison_rate * 1.5  # Allow some variance
        assert len(poisoned_data) == len(self.test_data)
        
        # Check that labels were flipped
        for record in poisoned_data:
            if record.custom_fields.get('poisoned', False):
                original_type = record.custom_fields.get('original_type')
                assert original_type is not None
                assert record.anomaly_type.value != original_type
    
    def test_backdoor_attack(self):
        """Test backdoor poisoning attack"""
        trigger_pattern = {'cpu_utilization': 85.5, 'memory_spike_kb': 256000}
        poisoned_data = self.generator.generate_backdoor_attack(self.test_data, trigger_pattern)
        
        backdoor_count = sum(1 for record in poisoned_data 
                           if record.custom_fields.get('backdoor', False))
        
        assert backdoor_count > 0
        assert backdoor_count <= len(self.test_data) * self.generator.max_poison_rate * 1.5
        
        # Check trigger pattern applied
        for record in poisoned_data:
            if record.custom_fields.get('backdoor', False):
                assert record.cpu_utilization == trigger_pattern['cpu_utilization']
                assert record.memory_spike_kb == trigger_pattern['memory_spike_kb']
                assert record.anomaly_type == AnomalyType.ADVERSARIAL_INJECTION
    
    def test_feature_corruption(self):
        """Test feature corruption attack"""
        corrupted_data = self.generator.generate_feature_corruption(self.test_data, 0.3)
        
        corrupted_count = sum(1 for record in corrupted_data 
                            if record.custom_fields.get('feature_corrupted', False))
        
        assert corrupted_count > 0
        assert corrupted_count <= len(self.test_data) * self.generator.max_poison_rate * 1.5
    
    def test_poison_rate_limits(self):
        """Test that poison rate limits are enforced"""
        with pytest.raises(ValueError):
            self.generator.generate_label_flip_attack(self.test_data, 0.1)  # Exceeds max rate


class TestEconomicAttackSimulator:
    """Test economic attack simulation"""
    
    def setup_method(self):
        """Setup for each test"""
        config = TestFixtures.create_test_config()
        self.simulator = EconomicAttackSimulator(config)
    
    def test_dow_attack_simulation(self):
        """Test Denial of Wallet attack simulation"""
        result = self.simulator.simulate_denial_of_wallet_attack(duration_minutes=5, intensity="medium")
        
        assert result.attack_type == AttackType.DENIAL_OF_WALLET
        assert result.end_time > result.start_time
        assert result.economic_impact > 0
        assert len(result.generated_telemetry) > 0
        
        # Check telemetry characteristics
        for telemetry in result.generated_telemetry:
            assert telemetry.custom_fields['attack_type'] == 'denial_of_wallet'
            assert telemetry.anomaly_type == AnomalyType.BILLING_ABUSE
    
    def test_cryptomining_simulation(self):
        """Test cryptomining attack simulation"""
        result = self.simulator.simulate_cryptomining_attack(duration_minutes=10)
        
        assert result.attack_type == AttackType.CRYPTOMINING
        assert result.economic_impact > 0
        assert len(result.generated_telemetry) > 0
        
        # Check mining characteristics
        for telemetry in result.generated_telemetry:
            assert telemetry.cpu_utilization > 90  # High CPU usage
            assert telemetry.duration > 200  # Long duration
            assert telemetry.custom_fields['attack_type'] == 'cryptomining'
    
    def test_billing_amplification(self):
        """Test billing amplification attack"""
        result = self.simulator.simulate_billing_amplification_attack()
        
        assert result.attack_type == AttackType.BILLING_AMPLIFICATION
        assert result.economic_impact > 0
        assert len(result.generated_telemetry) > 0
        
        # Check resource maximization
        for telemetry in result.generated_telemetry:
            assert telemetry.memory_spike_kb >= 3000000  # 3GB
            assert telemetry.cpu_utilization > 85
    
    def test_cost_calculation_accuracy(self):
        """Test accuracy of cost calculations"""
        result = self.simulator.simulate_denial_of_wallet_attack(duration_minutes=1, intensity="low")
        
        # Manual cost calculation for verification
        expected_cost = 0
        for telemetry in result.generated_telemetry:
            request_cost = 0.0000002  # Per request
            memory_gb = telemetry.memory_spike_kb / (1024 * 1024)
            compute_cost = memory_gb * telemetry.duration * 0.0000166667
            expected_cost += request_cost + compute_cost
        
        # Allow for small floating point differences
        assert abs(result.economic_impact - expected_cost) < 0.001


@pytest.mark.asyncio
class TestAdversarialAnomalyEngine:
    """Test main adversarial anomaly engine"""
    
    async def setup_method(self):
        """Setup for each test"""
        self.config = TestFixtures.create_test_config()
        self.engine = AdversarialAnomalyEngine(self.config)
        self.sample_telemetry = TestFixtures.create_sample_telemetry()
    
    async def test_engine_initialization(self):
        """Test engine initialization"""
        assert self.engine.config == self.config
        assert isinstance(self.engine.evasion_techniques, EvasionTechniques)
        assert isinstance(self.engine.poisoning_generator, PoisoningAttackGenerator)
        assert isinstance(self.engine.economic_simulator, EconomicAttackSimulator)
    
    async def test_evasion_attack_generation(self):
        """Test evasion attack generation"""
        attack_result = await self.engine.generate_adversarial_anomaly(
            self.sample_telemetry, AttackType.NOISE_INJECTION
        )
        
        assert isinstance(attack_result, AttackResult)
        assert attack_result.attack_type == AttackType.NOISE_INJECTION
        assert len(attack_result.generated_telemetry) > 0
        assert attack_result.perturbation_magnitude >= 0
        assert 0 <= attack_result.stealth_score <= 1
    
    async def test_economic_attack_generation(self):
        """Test economic attack generation"""
        attack_result = await self.engine.generate_adversarial_anomaly(
            self.sample_telemetry, AttackType.DENIAL_OF_WALLET
        )
        
        assert attack_result.attack_type == AttackType.DENIAL_OF_WALLET
        assert attack_result.economic_impact > 0
        assert len(attack_result.generated_telemetry) > 0
    
    async def test_poisoning_attack_generation(self):
        """Test poisoning attack generation"""
        attack_result = await self.engine.generate_adversarial_anomaly(
            self.sample_telemetry, AttackType.LABEL_FLIP
        )
        
        assert attack_result.attack_type == AttackType.LABEL_FLIP
        assert len(attack_result.generated_telemetry) > 0
        
        # Check for poisoning markers
        poisoned_count = sum(1 for record in attack_result.generated_telemetry
                           if record.custom_fields.get('poisoned', False))
        assert poisoned_count > 0
    
    async def test_attack_history_tracking(self):
        """Test attack history tracking"""
        initial_count = len(self.engine.attack_history)
        
        await self.engine.generate_adversarial_anomaly(
            self.sample_telemetry, AttackType.NOISE_INJECTION
        )
        
        assert len(self.engine.attack_history) == initial_count + 1
        assert isinstance(self.engine.attack_history[-1], AttackResult)
    
    async def test_attack_statistics(self):
        """Test attack statistics generation"""
        # Generate several attacks
        for attack_type in [AttackType.NOISE_INJECTION, AttackType.GRADIENT_MASKING, 
                           AttackType.DENIAL_OF_WALLET]:
            await self.engine.generate_adversarial_anomaly(self.sample_telemetry, attack_type)
        
        stats = self.engine.get_attack_statistics()
        
        assert stats['total_attacks'] >= 3
        assert 'attack_types' in stats
        assert 'success_rates' in stats
        assert 'average_stealth' in stats
        assert 'economic_impact' in stats
    
    async def test_performance_evaluation(self):
        """Test attack performance evaluation"""
        
        # Mock detection function
        def mock_detection_function(telemetry):
            # Simple mock: detect high CPU usage
            return {'anomaly_detected': telemetry.cpu_utilization > 80}
        
        test_data = TestFixtures.create_test_dataset(10)
        
        # Test with a subset of attack types for speed
        attack_types = [AttackType.NOISE_INJECTION, AttackType.CRYPTOMINING]
        
        for attack_type in attack_types:
            # Generate a few attacks to build some history
            for i in range(3):
                await self.engine.generate_adversarial_anomaly(test_data[i], attack_type)
        
        # Get performance report
        report = self.engine.get_attack_effectiveness_report()
        
        assert 'generation_timestamp' in report
        assert 'overall_statistics' in report
        assert 'attack_type_analysis' in report
        assert report['total_attack_types_tested'] > 0


@pytest.mark.asyncio 
class TestMultiStepCampaignOrchestrator:
    """Test multi-step campaign orchestration"""
    
    async def setup_method(self):
        """Setup for each test"""
        config = TestFixtures.create_test_config()
        self.engine = AdversarialAnomalyEngine(config)
        self.orchestrator = MultiStepCampaignOrchestrator(self.engine)
        self.sample_telemetry = TestFixtures.create_sample_telemetry()
    
    async def test_campaign_template_initialization(self):
        """Test campaign template initialization"""
        templates = self.orchestrator.campaign_templates
        
        assert 'cryptomining_campaign' in templates
        assert 'data_exfiltration_campaign' in templates
        assert 'denial_of_wallet_campaign' in templates
        
        # Check template structure
        template = templates['cryptomining_campaign']
        assert 'phases' in template
        assert 'stealth_level' in template
        assert len(template['phases']) > 0
    
    async def test_campaign_execution(self):
        """Test campaign execution"""
        # Use a simpler campaign for testing
        campaign_result = await self.orchestrator.execute_campaign(
            'cryptomining_campaign', self.sample_telemetry
        )
        
        assert campaign_result['campaign_type'] == 'cryptomining_campaign'
        assert 'phases' in campaign_result
        assert len(campaign_result['phases']) > 0
        assert 'total_telemetry' in campaign_result
        assert 'economic_impact' in campaign_result
        assert 'stealth_assessment' in campaign_result
    
    async def test_campaign_stealth_assessment(self):
        """Test campaign stealth assessment"""
        campaign_result = await self.orchestrator.execute_campaign(
            'data_exfiltration_campaign', self.sample_telemetry
        )
        
        stealth_metrics = campaign_result['stealth_assessment']
        
        assert 'detection_avoidance' in stealth_metrics
        assert 'temporal_distribution' in stealth_metrics
        assert 'attack_diversity' in stealth_metrics
        assert 'overall_stealth' in stealth_metrics
        
        # Check value ranges
        for metric_name, value in stealth_metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} should be between 0 and 1"
    
    async def test_invalid_campaign_type(self):
        """Test handling of invalid campaign type"""
        with pytest.raises(ValueError):
            await self.orchestrator.execute_campaign(
                'invalid_campaign', self.sample_telemetry
            )


class TestAdversarialValidationFramework:
    """Test adversarial validation framework"""
    
    def setup_method(self):
        """Setup for each test"""
        self.framework = AdversarialValidationFramework()
        self.baseline_data = TestFixtures.create_test_dataset(30)
        
        # Create a mock attack result
        self.attack_result = AttackResult(
            attack_id="test_attack_001",
            attack_type=AttackType.NOISE_INJECTION,
            start_time=time.time(),
            end_time=time.time() + 1,
            evasion_success=True,
            perturbation_magnitude=0.1,
            generated_telemetry=[TestFixtures.create_sample_telemetry()],
            stealth_score=0.7
        )
    
    def test_attack_realism_validation(self):
        """Test attack realism validation"""
        validation_scores = self.framework.validate_attack_realism(
            self.attack_result, self.baseline_data
        )
        
        required_metrics = [
            'statistical_realism',
            'temporal_consistency', 
            'resource_feasibility',
            'behavioral_plausibility',
            'overall_realism'
        ]
        
        for metric in required_metrics:
            assert metric in validation_scores
            assert 0.0 <= validation_scores[metric] <= 1.0
    
    def test_statistical_realism_validation(self):
        """Test statistical realism validation"""
        # Create adversarial data similar to baseline
        adversarial_data = []
        for record in self.baseline_data[:5]:
            modified = TestFixtures.create_sample_telemetry()
            modified.duration = record.duration + random.uniform(-0.1, 0.1)
            modified.memory_spike_kb = record.memory_spike_kb + random.randint(-5000, 5000)
            adversarial_data.append(modified)
        
        self.attack_result.generated_telemetry = adversarial_data
        
        scores = self.framework.validate_attack_realism(self.attack_result, self.baseline_data)
        
        # Statistical realism should be reasonably high for similar data
        assert scores['statistical_realism'] > 0.3
    
    def test_resource_feasibility_validation(self):
        """Test resource feasibility validation"""
        # Create infeasible attack result
        infeasible_telemetry = TestFixtures.create_sample_telemetry()
        infeasible_telemetry.duration = 1000  # 16+ minutes (exceeds Lambda limit)
        infeasible_telemetry.memory_spike_kb = 20000000  # 20GB (exceeds Lambda limit)
        infeasible_telemetry.cpu_utilization = 150  # Invalid percentage
        
        self.attack_result.generated_telemetry = [infeasible_telemetry]
        
        scores = self.framework.validate_attack_realism(self.attack_result, self.baseline_data)
        
        # Resource feasibility should be low
        assert scores['resource_feasibility'] < 0.5
    
    def test_temporal_consistency_validation(self):
        """Test temporal consistency validation"""
        # Create temporally inconsistent data
        inconsistent_data = []
        base_time = time.time()
        
        for i in range(5):
            record = TestFixtures.create_sample_telemetry()
            record.timestamp = base_time - i * 60  # Reverse chronological order
            inconsistent_data.append(record)
        
        self.attack_result.generated_telemetry = inconsistent_data
        
        scores = self.framework.validate_attack_realism(self.attack_result, self.baseline_data)
        
        # Temporal consistency should be low for reverse-ordered timestamps
        assert scores['temporal_consistency'] < 0.8


class TestAdversarialRobustnessAnalyzer:
    """Test adversarial robustness analyzer"""
    
    def setup_method(self):
        """Setup for each test"""
        self.analyzer = AdversarialRobustnessAnalyzer()
        
        # Create mock test results
        self.test_results = {
            'attack_results': {
                'noise_injection': {
                    'total_tests': 10,
                    'successful_attacks': 3,
                    'failed_attacks': 7,
                    'detection_rate': 0.7,
                    'average_perturbation': 0.15,
                    'economic_impact': 25.0
                },
                'cryptomining': {
                    'total_tests': 8,
                    'successful_attacks': 6,
                    'failed_attacks': 2,
                    'detection_rate': 0.25,
                    'average_perturbation': 0.05,
                    'economic_impact': 150.0
                }
            }
        }
        
        self.baseline_performance = {
            'detection_accuracy': 0.9,
            'false_positive_rate': 0.05
        }
    
    def test_comprehensive_analysis(self):
        """Test comprehensive robustness analysis"""
        analysis = self.analyzer.comprehensive_robustness_analysis(
            self.test_results, self.baseline_performance
        )
        
        required_sections = [
            'robustness_scores',
            'vulnerability_assessment',
            'improvement_recommendations',
            'benchmark_comparison',
            'risk_assessment'
        ]
        
        for section in required_sections:
            assert section in analysis
    
    def test_vulnerability_assessment(self):
        """Test vulnerability assessment"""
        # Create robustness scores that should trigger vulnerabilities
        robustness_scores = {
            'noise_injection': {
                'overall_robustness': 0.7,  # Medium risk
                'detection_robustness': 0.6
            },
            'cryptomining': {
                'overall_robustness': 0.2,  # Critical vulnerability
                'detection_robustness': 0.3
            }
        }
        
        vulnerabilities = self.analyzer._assess_vulnerabilities(robustness_scores)
        
        assert 'critical_vulnerabilities' in vulnerabilities
        assert 'high_risk_attacks' in vulnerabilities
        assert 'overall_risk_level' in vulnerabilities
        
        # Should identify cryptomining as critical vulnerability
        critical_attacks = [v['attack_type'] for v in vulnerabilities['critical_vulnerabilities']]
        assert 'cryptomining' in critical_attacks
    
    def test_improvement_recommendations(self):
        """Test improvement recommendation generation"""
        robustness_scores = {
            'noise_injection': {
                'detection_robustness': 0.4,  # Poor detection
                'performance_degradation': 0.8
            }
        }
        
        vulnerabilities = {
            'critical_vulnerabilities': [
                {'attack_type': 'noise_injection', 'robustness_score': 0.4}
            ],
            'vulnerability_patterns': {
                'detection_weakness': True,
                'performance_weakness': False
            }
        }
        
        recommendations = self.analyzer._generate_improvement_recommendations(
            robustness_scores, vulnerabilities
        )
        
        assert len(recommendations) > 0
        
        # Check for specific recommendation types
        rec_types = [rec['type'] for rec in recommendations]
        assert 'detection_improvement' in rec_types or 'general_detection' in rec_types
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison"""
        robustness_scores = {
            'noise_injection': {'overall_robustness': 0.9},  # Above benchmark
            'cryptomining': {'overall_robustness': 0.5}      # Below benchmark
        }
        
        comparison = self.analyzer._compare_to_benchmarks(robustness_scores)
        
        assert 'above_benchmark' in comparison
        assert 'below_benchmark' in comparison
        assert 'relative_performance' in comparison
        
        assert 'noise_injection' in comparison['above_benchmark']
        assert 'cryptomining' in comparison['below_benchmark']
        
        # Check that noise_injection is in above_benchmark list
        above_benchmark_types = [item['attack_type'] for item in comparison['above_benchmark']]
        assert 'noise_injection' in above_benchmark_types
    
    def test_operational_risk_assessment(self):
        """Test operational risk assessment"""
        robustness_scores = {
            'denial_of_wallet': {
                'overall_robustness': 0.3,  # High attack success probability
                'detection_robustness': 0.4
            }
        }
        
        test_results_with_economic = {
            'attack_results': {
                'denial_of_wallet': {
                    'economic_impact': 500.0  # High economic impact
                }
            }
        }
        
        risk_assessment = self.analyzer._assess_operational_risk(
            robustness_scores, test_results_with_economic
        )
        
        assert 'attack_success_probability' in risk_assessment
        assert 'economic_impact_risk' in risk_assessment
        assert 'operational_risk_score' in risk_assessment
        assert 'risk_mitigation_priority' in risk_assessment
        
        # High attack success should result in high operational risk
        assert risk_assessment['operational_risk_score'] > 0.5


class TestAdversarialMetricsCollector:
    """Test adversarial metrics collection and analysis"""
    
    def setup_method(self):
        """Setup for each test"""
        self.collector = AdversarialMetricsCollector()
        
        # Create sample attack result and detection response
        self.attack_result = AttackResult(
            attack_id="metrics_test_001",
            attack_type=AttackType.NOISE_INJECTION,
            start_time=time.time(),
            end_time=time.time() + 2,
            evasion_success=True,
            stealth_score=0.8,
            economic_impact=15.0,
            perturbation_magnitude=0.12
        )
        
        self.detection_response = {
            'response_time_ms': 150,
            'confidence': 0.75,
            'false_positive_risk': 0.1
        }
    
    def test_metrics_recording(self):
        """Test metrics recording"""
        initial_count = len(self.collector.metrics_history)
        
        self.collector.record_attack_metrics(self.attack_result, self.detection_response)
        
        assert len(self.collector.metrics_history) == initial_count + 1
        
        recorded_metric = self.collector.metrics_history[-1]
        assert recorded_metric['attack_id'] == self.attack_result.attack_id
        assert recorded_metric['attack_type'] == self.attack_result.attack_type.value
        assert recorded_metric['attack_success'] == self.attack_result.evasion_success
        assert recorded_metric['defense_response_time'] == self.detection_response['response_time_ms']
    
    def test_baseline_updates(self):
        """Test performance baseline updates"""
        # Record multiple attacks of the same type
        for i in range(5):
            attack_result = AttackResult(
                attack_id=f"baseline_test_{i}",
                attack_type=AttackType.GRADIENT_MASKING,
                start_time=time.time(),
                end_time=time.time() + 1,
                evasion_success=(i % 2 == 0),  # Alternate success/failure
                stealth_score=0.6 + i * 0.05,
                perturbation_magnitude=0.1 + i * 0.02
            )
            
            self.collector.record_attack_metrics(attack_result, self.detection_response)
        
        # Check baseline calculation
        baseline = self.collector.performance_baselines['gradient_masking']
        
        assert baseline['sample_count'] == 5
        assert 0.0 <= baseline['success_rate'] <= 1.0
        assert baseline['avg_stealth'] > 0
        assert baseline['avg_perturbation'] > 0
    
    def test_research_report_generation(self):
        """Test research report generation"""
        # Generate some test data
        attack_types = [AttackType.NOISE_INJECTION, AttackType.CRYPTOMINING, AttackType.GRADIENT_MASKING]
        
        for i in range(15):
            attack_type = attack_types[i % len(attack_types)]
            attack_result = AttackResult(
                attack_id=f"report_test_{i}",
                attack_type=attack_type,
                start_time=time.time() - (15 - i) * 60,  # Spaced in time
                end_time=time.time() - (15 - i) * 60 + 30,
                evasion_success=random.choice([True, False]),
                stealth_score=random.uniform(0.3, 0.9),
                economic_impact=random.uniform(5, 50),
                perturbation_magnitude=random.uniform(0.05, 0.25)
            )
            
            self.collector.record_attack_metrics(attack_result, self.detection_response)
        
        report = self.collector.generate_research_report()
        
        assert 'summary' in report
        assert 'attack_type_analysis' in report
        assert 'performance_baselines' in report
        
        # Check summary statistics
        summary = report['summary']
        assert summary['total_attacks'] == 15
        assert summary['unique_attack_types'] == len(attack_types)
        assert 0.0 <= summary['overall_success_rate'] <= 1.0
    
    def test_temporal_trend_analysis(self):
        """Test temporal trend analysis"""
        # Create data with improving success rate over time
        for i in range(20):
            success_probability = min(0.9, 0.1 + i * 0.04)  # Improving over time
            attack_result = AttackResult(
                attack_id=f"trend_test_{i}",
                attack_type=AttackType.ADAPTIVE_PERTURBATION,
                start_time=time.time() - (20 - i) * 300,  # 5-minute intervals
                end_time=time.time() - (20 - i) * 300 + 60,
                evasion_success=random.random() < success_probability,
                stealth_score=random.uniform(0.5, 0.8)
            )
            
            self.collector.record_attack_metrics(attack_result, self.detection_response)
        
        report = self.collector.generate_research_report()
        temporal_trends = report['temporal_trends']
        
        if not temporal_trends.get('insufficient_data'):
            assert 'trend_direction' in temporal_trends
            assert 'rolling_success_rate' in temporal_trends
            # Should detect increasing trend
            assert temporal_trends['trend_direction'] == 'increasing'


@pytest.mark.asyncio
class TestAdversarialTestSuite:
    """Test comprehensive adversarial test suite"""
    
    async def setup_method(self):
        """Setup for each test"""
        config = TestFixtures.create_test_config()
        self.engine = AdversarialAnomalyEngine(config)
        self.test_suite = AdversarialTestSuite(self.engine)
        self.test_data = TestFixtures.create_test_dataset(20)
        
        # Mock target system
        self.mock_target_system = Mock()
        self.mock_target_system.side_effect = lambda telemetry: {
            'anomaly_detected': telemetry.cpu_utilization > 80 or telemetry.memory_spike_kb > 300000,
            'confidence': 0.8,
            'response_time_ms': 50
        }
    
    async def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation"""
        # Run evaluation with limited attack types for speed
        with patch.object(self.engine, '_is_attack_enabled') as mock_enabled:
            # Only enable a few attack types for testing
            def enabled_check(attack_type):
                return attack_type in [AttackType.NOISE_INJECTION, AttackType.CRYPTOMINING]
            mock_enabled.side_effect = enabled_check
            
            evaluation_results = await self.test_suite.run_comprehensive_evaluation(
                self.mock_target_system, self.test_data[:5]  # Small dataset for speed
            )
        
        assert 'start_time' in evaluation_results
        assert 'attack_results' in evaluation_results
        assert 'robustness_metrics' in evaluation_results
        assert 'recommendations' in evaluation_results
        assert 'total_duration' in evaluation_results
        
        # Check that some attacks were evaluated
        assert len(evaluation_results['attack_results']) > 0
    
    async def test_attack_type_evaluation(self):
        """Test evaluation of specific attack type"""
        attack_results = await self.test_suite._evaluate_attack_type(
            AttackType.NOISE_INJECTION, self.mock_target_system, self.test_data[:3]
        )
        
        assert attack_results['attack_type'] == AttackType.NOISE_INJECTION.value
        assert attack_results['total_tests'] == 3
        assert 'successful_attacks' in attack_results
        assert 'failed_attacks' in attack_results
        assert 'detection_rate' in attack_results
        assert len(attack_results['individual_results']) == 3
    
    async def test_robustness_metrics_calculation(self):
        """Test robustness metrics calculation"""
        # Create mock attack results
        mock_attack_results = {
            'noise_injection': {
                'detection_rate': 0.8,
                'individual_results': [
                    {'success': False, 'perturbation_magnitude': 0.1},
                    {'success': True, 'perturbation_magnitude': 0.15},
                    {'success': False, 'perturbation_magnitude': 0.08}
                ]
            },
            'cryptomining': {
                'detection_rate': 0.6,
                'economic_impact': 75.0,
                'individual_results': [
                    {'success': True, 'perturbation_magnitude': 0.05},
                    {'success': False, 'perturbation_magnitude': 0.12}
                ]
            }
        }
        
        metrics = self.test_suite._calculate_robustness_metrics(mock_attack_results)
        
        assert 'overall_robustness' in metrics
        assert 'evasion_resistance' in metrics
        assert 'detection_effectiveness' in metrics
        assert 'worst_case_vulnerability' in metrics
        
        # Check value ranges
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} should be normalized to [0,1]"
    
    async def test_recommendation_generation(self):
        """Test recommendation generation"""
        # Create robustness metrics that should trigger recommendations
        poor_robustness_metrics = {
            'overall_robustness': 0.4,
            'detection_effectiveness': 0.5,
            'evasion_resistance': 0.3,
            'economic_resilience': 0.6,
            'worst_case_vulnerability': 0.8
        }
        
        recommendations = self.test_suite._generate_recommendations(poor_robustness_metrics)
        
        assert len(recommendations) > 0
        
        # Should recommend improvements for poor robustness
        rec_text = ' '.join(recommendations)
        assert 'robustness' in rec_text.lower() or 'detection' in rec_text.lower()


class TestQueryFreeAttackEngine:
    """Test query-free attack engine"""
    
    def setup_method(self):
        """Setup for each test"""
        config = TestFixtures.create_test_config()
        self.engine = QueryFreeAttackEngine(config)
        self.training_data = TestFixtures.create_test_dataset(30)
        self.target_record = TestFixtures.create_sample_telemetry()
    
    def test_surrogate_model_building(self):
        """Test surrogate model building"""
        self.engine.build_surrogate_model(self.training_data)
        
        assert len(self.engine.feature_statistics) > 0
        assert 'mean' in self.engine.feature_statistics
        assert 'std' in self.engine.feature_statistics
        assert 'correlation_matrix' in self.engine.feature_statistics
        assert len(self.engine.surrogate_models) > 0
    
    def test_query_free_generation(self):
        """Test query-free adversarial generation"""
        # Build surrogate model first
        self.engine.build_surrogate_model(self.training_data)
        
        # Generate adversarial example
        adversarial_record = self.engine.generate_query_free_adversarial(self.target_record)
        
        assert adversarial_record.custom_fields.get('query_free_adversarial') is True
        assert adversarial_record.custom_fields.get('generation_method') is not None
        
        # Should be different from original
        assert (adversarial_record.duration != self.target_record.duration or
                adversarial_record.memory_spike_kb != self.target_record.memory_spike_kb or
                adversarial_record.cpu_utilization != self.target_record.cpu_utilization)
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        features = self.engine._extract_features(self.target_record)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 7  # Expected number of features
        assert features[0] == self.target_record.duration
        assert features[1] == self.target_record.memory_spike_kb / 1000.0
    
    def test_statistical_perturbation(self):
        """Test statistical perturbation"""
        # Build surrogate model first
        self.engine.build_surrogate_model(self.training_data)
        
        original_features = self.engine._extract_features(self.target_record)
        perturbed_features = self.engine._statistical_perturbation(original_features)
        
        assert len(perturbed_features) == len(original_features)
        
        # Should have some differences
        differences = np.abs(perturbed_features - original_features)
        assert np.sum(differences) > 0


class TestTransferAttackEngine:
    """Test transfer attack engine"""
    
    def setup_method(self):
        """Setup for each test"""
        config = TestFixtures.create_test_config()
        self.engine = TransferAttackEngine(config)
        self.source_record = TestFixtures.create_sample_telemetry()
    
    def test_transfer_attack_generation(self):
        """Test transfer attack generation"""
        adversarial_record = self.engine.generate_transfer_attack(
            self.source_record, target_model_type="test_model"
        )
        
        assert adversarial_record.custom_fields.get('transfer_attack') is True
        assert adversarial_record.custom_fields.get('target_model_type') == "test_model"
        assert 'transferability_score' in adversarial_record.custom_fields
        
        # Should be different from original
        assert (adversarial_record.duration != self.source_record.duration or
                adversarial_record.memory_spike_kb != self.source_record.memory_spike_kb)
    
    def test_multi_level_feature_extraction(self):
        """Test multi-level feature extraction"""
        low_features = self.engine._extract_low_level_features(self.source_record)
        mid_features = self.engine._extract_mid_level_features(self.source_record)
        high_features = self.engine._extract_high_level_features(self.source_record)
        
        # Check feature types
        assert 'duration' in low_features
        assert 'memory' in low_features
        
        assert 'resource_intensity' in mid_features
        assert 'cpu_efficiency' in mid_features
        
        assert 'anomaly_indicator' in high_features
        assert 'execution_complexity' in high_features
    
    def test_feature_perturbation(self):
        """Test feature perturbation at different levels"""
        original_features = {'test_feature': 1.0, 'another_feature': 0.5}
        
        low_perturbed = self.engine._perturb_features(original_features, "low")
        mid_perturbed = self.engine._perturb_features(original_features, "mid")
        high_perturbed = self.engine._perturb_features(original_features, "high")
        
        # Check that perturbations were applied
        assert low_perturbed['test_feature'] != original_features['test_feature']
        assert mid_perturbed['test_feature'] != original_features['test_feature']
        assert high_perturbed['test_feature'] != original_features['test_feature']
    
    def test_transferability_score_calculation(self):
        """Test transferability score calculation"""
        original_features = {'feature1': 1.0, 'feature2': 2.0}
        
        # Small perturbation should give high transferability
        small_perturbed = {'feature1': 1.05, 'feature2': 2.1}
        high_score = self.engine._calculate_transferability_score(original_features, small_perturbed)
        
        # Large perturbation should give low transferability
        large_perturbed = {'feature1': 2.0, 'feature2': 4.0}
        low_score = self.engine._calculate_transferability_score(original_features, large_perturbed)
        
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0


@pytest.mark.asyncio
class TestPerformanceAndScalability:
    """Test performance and scalability aspects"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = TestFixtures.create_test_config()
        self.engine = AdversarialAnomalyEngine(self.config)  # Add this line
        self.large_dataset_size = 1000
        self.concurrent_requests = 50
    
    async def test_attack_generation_performance(self):
        """Test attack generation performance"""
        sample_telemetry = TestFixtures.create_sample_telemetry()
        
        start_time = time.time()
        
        # Generate multiple attacks quickly
        for _ in range(10):
            await self.engine.generate_adversarial_anomaly(
                sample_telemetry, AttackType.NOISE_INJECTION
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 10 attacks in reasonable time (< 5 seconds)
        assert total_time < 5.0, f"10 attacks took {total_time:.2f} seconds"
        
        # Average time per attack should be reasonable
        avg_time_per_attack = total_time / 10
        assert avg_time_per_attack < 0.5, f"Average time per attack: {avg_time_per_attack:.3f}s"
    
    async def test_concurrent_attack_generation(self):
        """Test concurrent attack generation"""
        sample_telemetry = TestFixtures.create_sample_telemetry()
        
        # Create multiple attack tasks
        attack_tasks = []
        for i in range(5):
            task = self.engine.generate_adversarial_anomaly(
                sample_telemetry, AttackType.GRADIENT_MASKING
            )
            attack_tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*attack_tasks)
        end_time = time.time()
        
        # All attacks should complete successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, AttackResult)
        
        # Concurrent execution should be faster than sequential
        total_time = end_time - start_time
        assert total_time < 2.0, f"Concurrent attacks took {total_time:.2f} seconds"

    def test_memory_usage_with_large_datasets(self):
        """Test memory usage with large datasets"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            large_dataset = TestFixtures.create_test_dataset(500)
            
            poisoning_generator = PoisoningAttackGenerator(0.05)
            poisoned_data = poisoning_generator.generate_label_flip_attack(large_dataset, 0.02)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            assert memory_increase < 100, f"Memory increase: {memory_increase:.1f} MB"
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")

    def test_attack_history_scalability(self):
        """Test attack history scalability"""
        # Generate many attack results
        for i in range(1000):
            attack_result = AttackResult(
                attack_id=f"scale_test_{i}",
                attack_type=AttackType.NOISE_INJECTION,
                start_time=time.time(),
                end_time=time.time() + 1,
                evasion_success=(i % 2 == 0)
            )
            self.engine.attack_history.append(attack_result)
        
        # Statistics generation should still be fast
        start_time = time.time()
        stats = self.engine.get_attack_statistics()
        stats_time = time.time() - start_time
        
        assert stats_time < 1.0, f"Statistics generation took {stats_time:.3f} seconds"
        assert stats['total_attacks'] == 1000


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Setup for error handling tests"""
        self.config = TestFixtures.create_test_config()
        self.engine = AdversarialAnomalyEngine(self.config)
    
    def test_invalid_attack_type_handling(self):
        """Test handling of invalid attack types"""
        sample_telemetry = TestFixtures.create_sample_telemetry()
        
        # Test with mock invalid attack type
        with patch.object(self.engine, '_is_attack_enabled', return_value=False):
            # This should use fallback mechanism
            task = self.engine.generate_adversarial_anomaly(sample_telemetry, AttackType.NOISE_INJECTION)
            # Should not raise exception
            assert asyncio.iscoroutine(task)
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets"""
        poisoning_generator = PoisoningAttackGenerator(0.05)
        
        # Empty dataset should not crash
        result = poisoning_generator.generate_label_flip_attack([], 0.02)
        assert result == []
        
        # Single item dataset
        single_item = [TestFixtures.create_sample_telemetry()]
        result = poisoning_generator.generate_label_flip_attack(single_item, 0.5)
        assert len(result) == 1
    
    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values"""
        sample_telemetry = TestFixtures.create_sample_telemetry()
        
        # Extreme noise levels
        extreme_noise_result = EvasionTechniques.noise_injection(sample_telemetry, noise_level=2.0)
        assert extreme_noise_result.duration > 0
        assert 0 <= extreme_noise_result.cpu_utilization <= 100
        
        # Zero noise level
        zero_noise_result = EvasionTechniques.noise_injection(sample_telemetry, noise_level=0.0)
        # Should be very similar to original (allowing for floating point precision)
        assert abs(zero_noise_result.duration - sample_telemetry.duration) < 0.001
    
    def test_malformed_telemetry_handling(self):
        """Test handling of malformed telemetry records"""
        # Create telemetry with extreme values
        malformed_telemetry = TelemetryRecord(
            event_id="malformed_001",
            timestamp=time.time(),
            function_id="",  # Empty function ID
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=AnomalyType.BENIGN,
            duration=-1.0,  # Negative duration
            memory_spike_kb=-1000,  # Negative memory
            cpu_utilization=150.0,  # Invalid CPU percentage
            network_io_bytes=-500,  # Negative network IO
            fallback_mode=False,
            source=TelemetrySource.PRIMARY,
            concurrency_id="test_concurrency_001"
        )
        
        # Evasion techniques should handle gracefully
        result = EvasionTechniques.noise_injection(malformed_telemetry, 0.1)
        
        # Should enforce valid ranges
        assert result.duration > 0
        assert result.memory_spike_kb >= 0
        assert 0 <= result.cpu_utilization <= 100
        assert result.network_io_bytes >= 0
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases"""
        # Minimal configuration
        minimal_config = AdversarialConfig(
            gan_latent_dim=10,  # Minimum allowed
            gan_feature_dim=5,
            gan_epochs=1,
            max_evasion_budget=0.01
        )
        
        issues = minimal_config.validate()
        assert len(issues) == 0  # Should be valid
        
        # Create engine with minimal config
        minimal_engine = AdversarialAnomalyEngine(minimal_config)
        assert minimal_engine.config == minimal_config


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_adversarial_workflow(self):
        """Test complete adversarial testing workflow"""
        # Setup
        config = TestFixtures.create_test_config()
        engine = AdversarialAnomalyEngine(config)
        test_suite = AdversarialTestSuite(engine)
        metrics_collector = AdversarialMetricsCollector()
        
        # Generate test data
        test_data = TestFixtures.create_test_dataset(10)
        
        # Mock target system
        def mock_detection_system(telemetry):
            # Simple heuristic detection
            anomaly_score = 0.0
            if telemetry.cpu_utilization > 80:
                anomaly_score += 0.4
            if telemetry.memory_spike_kb > 300000:
                anomaly_score += 0.3
            if telemetry.duration > 5.0:
                anomaly_score += 0.3
            
            return {
                'anomaly_detected': anomaly_score > 0.5,
                'confidence': min(1.0, anomaly_score),
                'response_time_ms': random.randint(10, 100)
            }
        
        # Generate attacks and collect metrics
        for i, sample in enumerate(test_data[:5]):  # Limit for test speed
            # Generate attack
            attack_type = [AttackType.NOISE_INJECTION, AttackType.CRYPTOMINING][i % 2]
            attack_result = await engine.generate_adversarial_anomaly(sample, attack_type)
            
            # Test against detection system
            for telemetry in attack_result.generated_telemetry:
                detection_response = mock_detection_system(telemetry)
                metrics_collector.record_attack_metrics(attack_result, detection_response)
        
        # Generate comprehensive reports
        effectiveness_report = engine.get_attack_effectiveness_report()
        research_report = metrics_collector.generate_research_report()
        
        # Verify reports contain expected data
        assert effectiveness_report['total_attack_types_tested'] > 0
        assert research_report['summary']['total_attacks'] > 0
        
        # Test robustness analysis
        analyzer = AdversarialRobustnessAnalyzer()
        mock_test_results = {
            'attack_results': {
                attack_type: {
                    'detection_rate': 0.6,
                    'economic_impact': 25.0,
                    'individual_results': [{'success': True}]
                }
                for attack_type in ['noise_injection', 'cryptomining']
            }
        }
        
        baseline_performance = {'detection_accuracy': 0.9, 'false_positive_rate': 0.05}
        robustness_analysis = analyzer.comprehensive_robustness_analysis(
            mock_test_results, baseline_performance
        )
        
        assert 'robustness_scores' in robustness_analysis
        assert 'recommendations' in robustness_analysis
    
    @pytest.mark.asyncio
    async def test_campaign_orchestration_workflow(self):
        """Test multi-step campaign orchestration workflow"""
        # Setup
        config = TestFixtures.create_test_config()
        engine = AdversarialAnomalyEngine(config)
        orchestrator = MultiStepCampaignOrchestrator(engine)
        
        sample_telemetry = TestFixtures.create_sample_telemetry()
        
        # Execute campaign
        campaign_result = await orchestrator.execute_campaign(
            'data_exfiltration_campaign', sample_telemetry
        )
        
        # Verify campaign execution
        assert campaign_result['campaign_type'] == 'data_exfiltration_campaign'
        assert len(campaign_result['phases']) > 0
        assert len(campaign_result['total_telemetry']) > 0
        
        # Verify phase execution
        for phase in campaign_result['phases']:
            assert 'phase_type' in phase
            assert 'telemetry' in phase
            assert 'success' in phase
        
        # Verify stealth assessment
        stealth_assessment = campaign_result['stealth_assessment']
        assert all(0.0 <= score <= 1.0 for score in stealth_assessment.values())


if __name__ == "__main__":
    # Run tests with appropriate markers
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not integration",  # Skip integration tests by default
        "--disable-warnings"
    ])
    
    # To run integration tests:
    # pytest test_adversarial.py -v -m integration
    
    # To run performance tests:
    # pytest test_adversarial.py -v -k "performance"
    
    # To run all tests:
    # pytest test_adversarial.py -v