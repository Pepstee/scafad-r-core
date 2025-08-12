#!/usr/bin/env python3
"""
SCAFAD Academic Evaluation Framework
===================================

Comprehensive experimental validation framework for academic publication.
Implements baseline comparisons, statistical validation, and reproducible experiments.

Academic Standards:
- Baseline comparisons with state-of-the-art methods
- Statistical significance testing
- Precision/Recall/F1 metrics
- ROC/AUC analysis
- Ablation studies
- Cross-validation
- Reproducible experimental setup

References:
- "The Elements of Statistical Learning" (Hastie et al., 2009)
- "Machine Learning: A Probabilistic Perspective" (Murphy, 2012)
- "Pattern Recognition and Machine Learning" (Bishop, 2006)
"""

import time
import logging
import numpy as np
import json
import statistics
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

# Statistical analysis
try:
    from scipy import stats
    from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu
    from sklearn.metrics import (
        precision_recall_fscore_support, roc_auc_score, roc_curve,
        confusion_matrix, classification_report
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_SKLEARN_AVAILABLE = False

# Import SCAFAD components
from layer0_core import AnomalyDetectionEngine, DetectionConfig
from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
from app_config import Layer0Config

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalResult:
    """Results from a single experimental run"""
    method_name: str
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    accuracy: float
    processing_time_ms: float
    memory_usage_mb: float
    false_positive_rate: float
    false_negative_rate: float
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class StatisticalValidation:
    """Statistical significance test results"""
    method_a: str
    method_b: str
    metric: str
    p_value: float
    effect_size: float
    significant: bool
    test_type: str


@dataclass 
class AblationResult:
    """Results from ablation study"""
    component_removed: str
    baseline_performance: float
    degraded_performance: float
    performance_drop: float
    relative_importance: float


class AcademicEvaluationFramework:
    """Comprehensive evaluation framework for academic validation"""
    
    def __init__(self, config: Optional[Layer0Config] = None):
        self.config = config or Layer0Config()
        self.scafad_engine = AnomalyDetectionEngine(DetectionConfig())
        
        # Initialize baseline methods
        if SCIPY_SKLEARN_AVAILABLE:
            self.baseline_methods = {
                'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
                'one_class_svm': OneClassSVM(nu=0.1),
                'local_outlier_factor': LocalOutlierFactor(contamination=0.1, novelty=True)
            }
        else:
            self.baseline_methods = {}
            logger.warning("Scipy/sklearn not available - baseline comparisons disabled")
        
        # Experimental parameters
        self.random_seed = 42
        self.n_folds = 5  # Cross-validation folds
        self.alpha = 0.05  # Statistical significance level
        self.min_effect_size = 0.2  # Minimum meaningful effect size
        
    def generate_synthetic_dataset(self, n_samples: int = 1000, 
                                 anomaly_ratio: float = 0.1) -> Tuple[List[TelemetryRecord], List[int]]:
        """
        Generate synthetic dataset for evaluation
        
        Creates realistic serverless execution patterns with controlled anomalies.
        """
        np.random.seed(self.random_seed)
        
        records = []
        labels = []
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies
        
        # Generate normal records
        for i in range(n_normal):
            record = self._generate_normal_record(i)
            records.append(record)
            labels.append(0)  # Normal
        
        # Generate anomalous records
        for i in range(n_anomalies):
            record = self._generate_anomalous_record(n_normal + i)
            records.append(record)
            labels.append(1)  # Anomalous
        
        # Shuffle for random distribution
        indices = np.arange(len(records))
        np.random.shuffle(indices)
        
        shuffled_records = [records[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]
        
        return shuffled_records, shuffled_labels
    
    def _generate_normal_record(self, record_id: int) -> TelemetryRecord:
        """Generate a normal telemetry record"""
        return TelemetryRecord(
            event_id=f"normal_{record_id}",
            timestamp=time.time() + record_id,
            function_id="test_function",
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=AnomalyType.BENIGN,
            duration=np.random.normal(1.0, 0.3),  # Normal execution time
            memory_spike_kb=int(np.random.normal(512, 100)),  # Normal memory usage
            cpu_utilization=np.random.normal(25, 10),  # Normal CPU usage
            network_io_bytes=int(np.random.normal(1000, 300)),
            fallback_mode=False,
            source=TelemetrySource.SCAFAD_LAYER0,
            concurrency_id=f"normal_concurrency_{record_id}"
        )
    
    def _generate_anomalous_record(self, record_id: int) -> TelemetryRecord:
        """Generate an anomalous telemetry record"""
        anomaly_types = [
            AnomalyType.CPU_BURST,
            AnomalyType.MEMORY_SPIKE,
            AnomalyType.TIMEOUT_ANOMALY,
            AnomalyType.EXECUTION_FAILURE
        ]
        
        anomaly_type = np.random.choice(anomaly_types)
        
        # Create anomalous characteristics
        if anomaly_type == AnomalyType.CPU_BURST:
            duration = np.random.normal(1.0, 0.3)
            memory_spike_kb = int(np.random.normal(512, 100))
            cpu_utilization = np.random.normal(85, 10)  # High CPU
        elif anomaly_type == AnomalyType.MEMORY_SPIKE:
            duration = np.random.normal(1.0, 0.3)
            memory_spike_kb = int(np.random.normal(2048, 500))  # High memory
            cpu_utilization = np.random.normal(25, 10)
        elif anomaly_type == AnomalyType.TIMEOUT_ANOMALY:
            duration = np.random.normal(8.0, 2.0)  # Long execution
            memory_spike_kb = int(np.random.normal(512, 100))
            cpu_utilization = np.random.normal(25, 10)
        else:  # EXECUTION_FAILURE
            duration = np.random.normal(0.1, 0.05)  # Very short
            memory_spike_kb = int(np.random.normal(512, 100))
            cpu_utilization = np.random.normal(5, 2)  # Low CPU
        
        return TelemetryRecord(
            event_id=f"anomaly_{record_id}",
            timestamp=time.time() + record_id,
            function_id="test_function",
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=anomaly_type,
            duration=max(0.01, duration),  # Ensure positive
            memory_spike_kb=max(1, int(memory_spike_kb)),  # Ensure positive
            cpu_utilization=max(0, min(100, cpu_utilization)),  # Clamp to [0,100]
            network_io_bytes=int(np.random.normal(1000, 300)),
            fallback_mode=False,
            source=TelemetrySource.SCAFAD_LAYER0,
            concurrency_id=f"anomaly_concurrency_{record_id}"
        )
    
    async def evaluate_scafad_performance(self, 
                                        records: List[TelemetryRecord], 
                                        true_labels: List[int]) -> ExperimentalResult:
        """Evaluate SCAFAD performance on dataset"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        predictions = []
        confidence_scores = []
        
        for record in records:
            result = self.scafad_engine.detect_anomalies(record)
            prediction = 1 if result.final_anomaly_detected else 0
            confidence = result.combined_confidence
            
            predictions.append(prediction)
            confidence_scores.append(confidence)
        
        processing_time = (time.time() - start_time) * 1000
        memory_usage = self._get_memory_usage() - start_memory
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        auc_score = roc_auc_score(true_labels, confidence_scores) if len(set(true_labels)) > 1 else 0.0
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        # Calculate error rates
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return ExperimentalResult(
            method_name="SCAFAD",
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
            accuracy=accuracy,
            processing_time_ms=processing_time,
            memory_usage_mb=memory_usage,
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
    
    def evaluate_baseline_method(self, method_name: str, 
                               records: List[TelemetryRecord], 
                               true_labels: List[int]) -> ExperimentalResult:
        """Evaluate baseline method performance"""
        if not SCIPY_SKLEARN_AVAILABLE or method_name not in self.baseline_methods:
            return ExperimentalResult(
                method_name=method_name,
                precision=0.0, recall=0.0, f1_score=0.0, auc_score=0.0,
                accuracy=0.0, processing_time_ms=0.0, memory_usage_mb=0.0,
                false_positive_rate=1.0, false_negative_rate=1.0
            )
        
        # Convert records to feature matrix
        features = self._extract_baseline_features(records)
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        method = self.baseline_methods[method_name]
        
        # Train and predict
        if method_name == 'local_outlier_factor':
            # LOF needs fit_predict
            predictions_raw = method.fit_predict(features)
            predictions = [1 if p == -1 else 0 for p in predictions_raw]
            confidence_scores = -method.negative_outlier_factor_
        else:
            method.fit(features[np.array(true_labels) == 0])  # Train on normal data only
            predictions_raw = method.predict(features)
            predictions = [1 if p == -1 else 0 for p in predictions_raw]
            confidence_scores = method.decision_function(features)
        
        processing_time = (time.time() - start_time) * 1000
        memory_usage = self._get_memory_usage() - start_memory
        
        # Normalize confidence scores to [0,1]
        confidence_scores = (confidence_scores - np.min(confidence_scores)) / (np.max(confidence_scores) - np.min(confidence_scores) + 1e-8)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        auc_score = roc_auc_score(true_labels, confidence_scores) if len(set(true_labels)) > 1 else 0.0
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        # Calculate error rates
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return ExperimentalResult(
            method_name=method_name,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
            accuracy=accuracy,
            processing_time_ms=processing_time,
            memory_usage_mb=memory_usage,
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
    
    def _extract_baseline_features(self, records: List[TelemetryRecord]) -> np.ndarray:
        """Extract features for baseline methods"""
        features = []
        
        for record in records:
            feature_vector = [
                record.duration,
                record.memory_spike_kb,
                record.cpu_utilization,
                record.network_io_bytes,
                float(record.fallback_mode),
                record.timestamp % 86400  # Time of day
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def perform_statistical_significance_test(self, 
                                            results_a: List[float], 
                                            results_b: List[float],
                                            method_a: str, 
                                            method_b: str,
                                            metric: str) -> StatisticalValidation:
        """Perform statistical significance test between two methods"""
        
        if not SCIPY_SKLEARN_AVAILABLE:
            return StatisticalValidation(
                method_a=method_a, method_b=method_b, metric=metric,
                p_value=1.0, effect_size=0.0, significant=False, test_type="unavailable"
            )
        
        # Choose appropriate test
        if len(results_a) == len(results_b):
            # Paired test (same dataset splits)
            statistic, p_value = wilcoxon(results_a, results_b, alternative='two-sided')
            test_type = "Wilcoxon signed-rank test"
        else:
            # Unpaired test
            statistic, p_value = mannwhitneyu(results_a, results_b, alternative='two-sided')
            test_type = "Mann-Whitney U test"
        
        # Calculate effect size (Cohen's d)
        mean_a, mean_b = np.mean(results_a), np.mean(results_b)
        std_a, std_b = np.std(results_a, ddof=1), np.std(results_b, ddof=1)
        pooled_std = np.sqrt(((len(results_a) - 1) * std_a**2 + (len(results_b) - 1) * std_b**2) / 
                           (len(results_a) + len(results_b) - 2))
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        return StatisticalValidation(
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            p_value=p_value,
            effect_size=abs(effect_size),
            significant=p_value < self.alpha and abs(effect_size) >= self.min_effect_size,
            test_type=test_type
        )
    
    def perform_ablation_study(self, 
                              records: List[TelemetryRecord], 
                              true_labels: List[int]) -> List[AblationResult]:
        """Perform ablation study to assess component importance"""
        
        # Get baseline performance with all components
        baseline_result = await self.evaluate_scafad_performance(records, true_labels)
        baseline_f1 = baseline_result.f1_score
        
        ablation_results = []
        
        # Test removing each component
        components_to_test = [
            'isolation_forest',
            'lstm_anomaly',
            'statistical_outlier',
            'graph_analysis',
            'economic_abuse',
            'correlation_break'
        ]
        
        for component in components_to_test:
            # Temporarily disable component
            original_algorithms = self.scafad_engine.algorithms.copy()
            if component in self.scafad_engine.algorithms:
                del self.scafad_engine.algorithms[component]
                
                # Re-normalize weights
                self.scafad_engine._normalize_algorithm_weights()
                
                # Evaluate degraded performance
                degraded_result = await self.evaluate_scafad_performance(records, true_labels)
                degraded_f1 = degraded_result.f1_score
                
                # Calculate importance
                performance_drop = baseline_f1 - degraded_f1
                relative_importance = performance_drop / baseline_f1 if baseline_f1 > 0 else 0.0
                
                ablation_results.append(AblationResult(
                    component_removed=component,
                    baseline_performance=baseline_f1,
                    degraded_performance=degraded_f1,
                    performance_drop=performance_drop,
                    relative_importance=relative_importance
                ))
                
                # Restore algorithms
                self.scafad_engine.algorithms = original_algorithms
                self.scafad_engine._normalize_algorithm_weights()
        
        return ablation_results
    
    async def run_comprehensive_evaluation(self, 
                                         dataset_sizes: List[int] = [500, 1000, 2000],
                                         anomaly_ratios: List[float] = [0.05, 0.1, 0.15]) -> Dict[str, Any]:
        """Run comprehensive evaluation across multiple datasets"""
        
        logger.info("Starting comprehensive academic evaluation...")
        
        all_results = defaultdict(list)
        statistical_tests = []
        ablation_studies = []
        
        for dataset_size in dataset_sizes:
            for anomaly_ratio in anomaly_ratios:
                logger.info(f"Evaluating dataset: size={dataset_size}, anomaly_ratio={anomaly_ratio}")
                
                # Generate dataset
                records, labels = self.generate_synthetic_dataset(dataset_size, anomaly_ratio)
                
                # Evaluate SCAFAD
                scafad_result = await self.evaluate_scafad_performance(records, labels)
                all_results['SCAFAD'].append(scafad_result)
                
                # Evaluate baselines
                for method_name in self.baseline_methods.keys():
                    baseline_result = self.evaluate_baseline_method(method_name, records, labels)
                    all_results[method_name].append(baseline_result)
                
                # Perform ablation study (once per dataset size)
                if anomaly_ratio == 0.1:  # Standard ratio
                    ablation_result = self.perform_ablation_study(records, labels)
                    ablation_studies.append({
                        'dataset_size': dataset_size,
                        'results': ablation_result
                    })
        
        # Statistical significance testing
        if SCIPY_SKLEARN_AVAILABLE:
            scafad_f1_scores = [r.f1_score for r in all_results['SCAFAD']]
            
            for method_name in self.baseline_methods.keys():
                baseline_f1_scores = [r.f1_score for r in all_results[method_name]]
                
                stat_test = self.perform_statistical_significance_test(
                    scafad_f1_scores, baseline_f1_scores,
                    'SCAFAD', method_name, 'F1-Score'
                )
                statistical_tests.append(stat_test)
        
        # Aggregate results
        aggregated_results = {}
        for method_name, results in all_results.items():
            aggregated_results[method_name] = {
                'mean_precision': np.mean([r.precision for r in results]),
                'mean_recall': np.mean([r.recall for r in results]),
                'mean_f1': np.mean([r.f1_score for r in results]),
                'mean_auc': np.mean([r.auc_score for r in results]),
                'mean_processing_time_ms': np.mean([r.processing_time_ms for r in results]),
                'std_f1': np.std([r.f1_score for r in results]),
                'results': results
            }
        
        return {
            'aggregated_results': aggregated_results,
            'statistical_tests': statistical_tests,
            'ablation_studies': ablation_studies,
            'experimental_parameters': {
                'dataset_sizes': dataset_sizes,
                'anomaly_ratios': anomaly_ratios,
                'random_seed': self.random_seed,
                'significance_level': self.alpha,
                'min_effect_size': self.min_effect_size
            }
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def generate_academic_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate academic-standard evaluation report"""
        
        report = []
        report.append("# SCAFAD Layer 0: Comprehensive Experimental Evaluation")
        report.append("=" * 60)
        report.append()
        
        # Executive summary
        report.append("## Executive Summary")
        report.append()
        
        scafad_results = evaluation_results['aggregated_results']['SCAFAD']
        report.append(f"SCAFAD achieved mean F1-score: {scafad_results['mean_f1']:.3f} ± {scafad_results['std_f1']:.3f}")
        report.append(f"Mean processing time: {scafad_results['mean_processing_time_ms']:.2f}ms")
        report.append()
        
        # Baseline comparisons
        report.append("## Baseline Comparisons")
        report.append()
        report.append("| Method | F1-Score | Precision | Recall | AUC | Processing Time (ms) |")
        report.append("|--------|----------|-----------|--------|-----|---------------------|")
        
        for method_name, results in evaluation_results['aggregated_results'].items():
            report.append(f"| {method_name} | {results['mean_f1']:.3f} | "
                         f"{results['mean_precision']:.3f} | {results['mean_recall']:.3f} | "
                         f"{results['mean_auc']:.3f} | {results['mean_processing_time_ms']:.2f} |")
        
        report.append()
        
        # Statistical significance
        report.append("## Statistical Significance Tests")
        report.append()
        
        for test in evaluation_results['statistical_tests']:
            significance = "✓ Significant" if test.significant else "✗ Not significant"
            report.append(f"**{test.method_a} vs {test.method_b}**: {significance}")
            report.append(f"- p-value: {test.p_value:.4f}")
            report.append(f"- Effect size: {test.effect_size:.3f}")
            report.append(f"- Test: {test.test_type}")
            report.append()
        
        # Ablation study
        report.append("## Ablation Study")
        report.append()
        report.append("Component importance analysis:")
        report.append()
        
        if evaluation_results['ablation_studies']:
            # Average across dataset sizes
            component_importance = defaultdict(list)
            for study in evaluation_results['ablation_studies']:
                for result in study['results']:
                    component_importance[result.component_removed].append(result.relative_importance)
            
            report.append("| Component | Relative Importance | Performance Drop |")
            report.append("|-----------|--------------------|--------------------|")
            
            for component, importances in component_importance.items():
                mean_importance = np.mean(importances)
                report.append(f"| {component} | {mean_importance:.3f} | {mean_importance*100:.1f}% |")
        
        report.append()
        report.append("## Experimental Parameters")
        report.append()
        params = evaluation_results['experimental_parameters']
        report.append(f"- Dataset sizes: {params['dataset_sizes']}")
        report.append(f"- Anomaly ratios: {params['anomaly_ratios']}")  
        report.append(f"- Random seed: {params['random_seed']}")
        report.append(f"- Significance level: {params['significance_level']}")
        report.append(f"- Minimum effect size: {params['min_effect_size']}")
        
        return "\\n".join(report)


# Example usage and self-test
async def run_academic_evaluation():
    """Run comprehensive academic evaluation"""
    
    logger.info("Initializing Academic Evaluation Framework...")
    
    framework = AcademicEvaluationFramework()
    
    # Run evaluation with smaller datasets for testing
    results = await framework.run_comprehensive_evaluation(
        dataset_sizes=[100, 200],  # Smaller for testing
        anomaly_ratios=[0.1, 0.15]
    )
    
    # Generate report
    report = framework.generate_academic_report(results)
    
    return results, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    results, report = asyncio.run(run_academic_evaluation())
    
    print("Academic Evaluation Complete!")
    print("=" * 50)
    print(report)