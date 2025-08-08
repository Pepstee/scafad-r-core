# telemetry/fetch_logs.py
"""
SCAFAD Layer 0: Enhanced Telemetry Harvester
============================================

Advanced log fetching and processing system for SCAFAD Layer 0 telemetry.
Integrates with the comprehensive telemetry controller and provides
multi-modal data extraction, analysis, and archival capabilities.

Version: v4.2-enhanced
Institution: Birmingham Newman University
Compatible with: Layer0_AdaptiveTelemetryController
"""

import json
import csv
import subprocess
import os
import time
import asyncio
import hashlib
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import re
import gzip

# Configuration
CONFIG = {
    'log_group': "/aws/lambda/scafad-test-stack-HelloWorldFunction-k79tX3iBcK74",
    'lookback_hours': 1,
    'max_events': 1000,
    'enable_compression': True,
    'enable_analysis': True,
    'export_formats': ['csv', 'json', 'parquet'],
    'telemetry_version': 'v4.2-enhanced',
    'archive_retention_days': 30
}

@dataclass
class TelemetryMetrics:
    """Comprehensive telemetry metrics for analysis"""
    total_events: int = 0
    structured_logs: int = 0
    side_traces: int = 0
    malformed_logs: int = 0
    anomaly_counts: Dict[str, int] = None
    fallback_events: int = 0
    economic_risk_events: int = 0
    processing_time_ms: float = 0.0
    data_quality_score: float = 0.0
    
    def __post_init__(self):
        if self.anomaly_counts is None:
            self.anomaly_counts = {}

class AdvancedTelemetryProcessor:
    """Advanced processor for SCAFAD Layer 0 telemetry data"""
    
    def __init__(self):
        self.metrics = TelemetryMetrics()
        self.graph_nodes = []
        self.provenance_chains = []
        self.economic_indicators = []
        self.temporal_patterns = defaultdict(list)
        
    def process_structured_log(self, log_entry: Dict) -> Dict:
        """Process structured telemetry logs with enhanced analysis"""
        
        # Validate against SCAFAD Layer 0 schema
        if not self._validate_scafad_schema(log_entry):
            return {'status': 'invalid', 'reason': 'schema_mismatch'}
        
        # Extract telemetry record components
        telemetry_data = {
            'event_id': log_entry.get('event_id'),
            'timestamp': log_entry.get('timestamp'),
            'function_id': log_entry.get('function_profile_id', log_entry.get('function_id')),
            'execution_phase': log_entry.get('execution_phase'),
            'anomaly_type': log_entry.get('anomaly_type'),
            'duration': log_entry.get('duration', 0),
            'memory_spike_kb': log_entry.get('memory_spike_kb', 0),
            'cpu_utilization': log_entry.get('cpu_utilization', 0),
            'network_io_bytes': log_entry.get('network_io_bytes', 0),
            'fallback_mode': log_entry.get('fallback_mode', False),
            'source': log_entry.get('source', 'unknown'),
            'concurrency_id': log_entry.get('concurrency_id'),
            'provenance_id': log_entry.get('provenance_id'),
            'graph_node_id': log_entry.get('graph_node_id'),
            'adversarial_score': log_entry.get('adversarial_score', 0.0),
            'economic_risk_score': log_entry.get('economic_risk_score', 0.0),
            'completeness_score': log_entry.get('completeness_score', 1.0),
            'log_version': log_entry.get('log_version', {})
        }
        
        # Enhanced analytics
        enhanced_data = self._enhance_telemetry_data(telemetry_data)
        
        # Update metrics
        self._update_metrics(enhanced_data)
        
        # Store temporal patterns
        self._store_temporal_pattern(enhanced_data)
        
        return enhanced_data
    
    def _validate_scafad_schema(self, log_entry: Dict) -> bool:
        """Validate log entry against SCAFAD Layer 0 schema"""
        required_fields = ['event_id', 'timestamp', 'anomaly_type']
        
        # Check required fields
        for field in required_fields:
            if field not in log_entry:
                return False
        
        # Check log version compatibility
        log_version = log_entry.get('log_version', {})
        if isinstance(log_version, dict):
            version = log_version.get('version', '')
            if version.startswith('v4.'):
                return True
        
        return False
    
    def _enhance_telemetry_data(self, telemetry_data: Dict) -> Dict:
        """Add enhanced analytics to telemetry data"""
        enhanced = telemetry_data.copy()
        
        # Calculate risk scores
        enhanced['composite_risk_score'] = self._calculate_composite_risk(telemetry_data)
        
        # Detect patterns
        enhanced['pattern_classification'] = self._classify_execution_pattern(telemetry_data)
        
        # Performance analytics
        enhanced['performance_percentile'] = self._calculate_performance_percentile(telemetry_data)
        
        # Behavioral fingerprint
        enhanced['behavioral_fingerprint'] = self._generate_behavioral_fingerprint(telemetry_data)
        
        # Economic impact
        enhanced['estimated_cost_impact'] = self._estimate_cost_impact(telemetry_data)
        
        return enhanced
    
    def _calculate_composite_risk(self, data: Dict) -> float:
        """Calculate composite risk score from multiple indicators"""
        risk_factors = []
        
        # Adversarial risk
        if data.get('adversarial_score', 0) > 0:
            risk_factors.append(data['adversarial_score'])
        
        # Economic risk
        if data.get('economic_risk_score', 0) > 0:
            risk_factors.append(data['economic_risk_score'])
        
        # Performance anomaly risk
        duration = data.get('duration', 0)
        if duration > 1.0:  # > 1 second
            risk_factors.append(min(duration / 10.0, 1.0))
        
        # Memory anomaly risk
        memory_kb = data.get('memory_spike_kb', 0)
        if memory_kb > 50000:  # > 50MB
            risk_factors.append(min(memory_kb / 100000.0, 1.0))
        
        # Fallback mode indicates risk
        if data.get('fallback_mode', False):
            risk_factors.append(0.8)
        
        return np.mean(risk_factors) if risk_factors else 0.0
    
    def _classify_execution_pattern(self, data: Dict) -> str:
        """Classify execution pattern based on telemetry characteristics"""
        duration = data.get('duration', 0)
        memory_kb = data.get('memory_spike_kb', 0)
        cpu_util = data.get('cpu_utilization', 0)
        anomaly_type = data.get('anomaly_type', 'benign')
        
        # Pattern classification logic
        if anomaly_type == 'cold_start':
            return 'cold_initialization'
        elif cpu_util > 80:
            return 'compute_intensive'
        elif memory_kb > 30000:
            return 'memory_intensive'
        elif duration > 5.0:
            return 'long_running'
        elif data.get('network_io_bytes', 0) > 10000:
            return 'io_intensive'
        else:
            return 'standard_execution'
    
    def _calculate_performance_percentile(self, data: Dict) -> float:
        """Calculate performance percentile based on historical data"""
        duration = data.get('duration', 0)
        
        # Simple percentile calculation (would use historical data in production)
        if duration < 0.1:
            return 95.0  # Very fast
        elif duration < 0.5:
            return 75.0  # Fast
        elif duration < 1.0:
            return 50.0  # Average
        elif duration < 2.0:
            return 25.0  # Slow
        else:
            return 5.0   # Very slow
    
    def _generate_behavioral_fingerprint(self, data: Dict) -> str:
        """Generate behavioral fingerprint hash"""
        fingerprint_data = {
            'execution_phase': data.get('execution_phase'),
            'pattern': self._classify_execution_pattern(data),
            'source': data.get('source'),
            'function_id': data.get('function_id')
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    
    def _estimate_cost_impact(self, data: Dict) -> float:
        """Estimate cost impact in USD (simplified calculation)"""
        duration = data.get('duration', 0)
        memory_kb = data.get('memory_spike_kb', 0)
        
        # AWS Lambda pricing approximation
        memory_gb = memory_kb / (1024 * 1024)
        gb_seconds = memory_gb * duration
        
        # Base cost per GB-second (approximate)
        base_cost = gb_seconds * 0.0000166667
        
        # Anomaly multipliers
        anomaly_type = data.get('anomaly_type', 'benign')
        if anomaly_type == 'cold_start':
            base_cost *= 1.5
        elif 'cpu_burst' in anomaly_type:
            base_cost *= 2.0
        elif data.get('fallback_mode', False):
            base_cost *= 3.0
        
        return base_cost
    
    def _update_metrics(self, data: Dict):
        """Update comprehensive metrics"""
        self.metrics.structured_logs += 1
        
        # Count anomaly types
        anomaly_type = data.get('anomaly_type', 'unknown')
        if anomaly_type not in self.metrics.anomaly_counts:
            self.metrics.anomaly_counts[anomaly_type] = 0
        self.metrics.anomaly_counts[anomaly_type] += 1
        
        # Count fallback events
        if data.get('fallback_mode', False):
            self.metrics.fallback_events += 1
        
        # Count economic risk events
        if data.get('economic_risk_score', 0) > 0.5:
            self.metrics.economic_risk_events += 1
    
    def _store_temporal_pattern(self, data: Dict):
        """Store temporal execution patterns for analysis"""
        timestamp = data.get('timestamp', time.time())
        function_id = data.get('function_id', 'unknown')
        
        pattern_key = f"{function_id}_{data.get('execution_phase', 'unknown')}"
        self.temporal_patterns[pattern_key].append({
            'timestamp': timestamp,
            'duration': data.get('duration', 0),
            'anomaly_type': data.get('anomaly_type'),
            'risk_score': data.get('composite_risk_score', 0)
        })

class TelemetryExporter:
    """Advanced telemetry data exporter with multiple format support"""
    
    def __init__(self, base_path: str = "telemetry"):
        self.base_path = base_path
        self.archive_path = os.path.join(base_path, "archive")
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.archive_path, exist_ok=True)
        os.makedirs(os.path.join(self.archive_path, "compressed"), exist_ok=True)
    
    def export_structured_logs(self, logs: List[Dict], timestamp: str) -> Dict[str, str]:
        """Export structured logs in multiple formats"""
        exports = {}
        
        if not logs:
            return exports
        
        # Determine fieldnames from all logs
        fieldnames = set()
        for log in logs:
            fieldnames.update(log.keys())
        fieldnames = sorted(fieldnames)
        
        # CSV Export
        if 'csv' in CONFIG['export_formats']:
            csv_path = self._export_csv(logs, fieldnames, timestamp)
            exports['csv'] = csv_path
        
        # JSON Export
        if 'json' in CONFIG['export_formats']:
            json_path = self._export_json(logs, timestamp)
            exports['json'] = json_path
        
        # Compressed exports
        if CONFIG['enable_compression']:
            compressed_exports = self._create_compressed_exports(exports)
            exports.update(compressed_exports)
        
        return exports
    
    def _export_csv(self, logs: List[Dict], fieldnames: List[str], timestamp: str) -> str:
        """Export logs to CSV format"""
        filename = f"lambda_telemetry_{timestamp}.csv"
        filepath = os.path.join(self.archive_path, filename)
        
        with open(filepath, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)
        
        # Also create current file
        current_path = os.path.join(self.base_path, "lambda_telemetry.csv")
        with open(current_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)
        
        return filepath
    
    def _export_json(self, logs: List[Dict], timestamp: str) -> str:
        """Export logs to JSON format"""
        filename = f"lambda_telemetry_{timestamp}.json"
        filepath = os.path.join(self.archive_path, filename)
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'record_count': len(logs),
                'scafad_version': CONFIG['telemetry_version'],
                'schema_version': 'v4.2'
            },
            'telemetry_records': logs
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Also create current file
        current_path = os.path.join(self.base_path, "lambda_telemetry.json")
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath
    
    def _create_compressed_exports(self, exports: Dict[str, str]) -> Dict[str, str]:
        """Create compressed versions of exports"""
        compressed = {}
        
        for format_type, filepath in exports.items():
            if os.path.exists(filepath):
                compressed_path = self._compress_file(filepath)
                compressed[f"{format_type}_compressed"] = compressed_path
        
        return compressed
    
    def _compress_file(self, filepath: str) -> str:
        """Compress a file using gzip"""
        compressed_filename = os.path.basename(filepath) + ".gz"
        compressed_path = os.path.join(self.archive_path, "compressed", compressed_filename)
        
        with open(filepath, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        return compressed_path
    
    def export_side_traces(self, traces: List[str], timestamp: str) -> str:
        """Export side channel traces"""
        filename = f"side_channel_trace_{timestamp}.log"
        filepath = os.path.join(self.archive_path, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(trace + "\n")
        
        # Also create current file
        current_path = os.path.join(self.base_path, "side_channel_trace.log")
        with open(current_path, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(trace + "\n")
        
        return filepath
    
    def export_analytics_report(self, processor: AdvancedTelemetryProcessor, timestamp: str) -> str:
        """Export comprehensive analytics report"""
        filename = f"analytics_report_{timestamp}.json"
        filepath = os.path.join(self.archive_path, filename)
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'scafad_version': CONFIG['telemetry_version'],
                'analysis_window_hours': CONFIG['lookback_hours']
            },
            'summary_metrics': asdict(processor.metrics),
            'temporal_analysis': self._analyze_temporal_patterns(processor.temporal_patterns),
            'risk_distribution': self._analyze_risk_distribution(processor),
            'performance_insights': self._generate_performance_insights(processor),
            'recommendations': self._generate_recommendations(processor)
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        return filepath
    
    def _analyze_temporal_patterns(self, patterns: Dict) -> Dict:
        """Analyze temporal execution patterns"""
        analysis = {}
        
        for pattern_key, events in patterns.items():
            if not events:
                continue
            
            durations = [e['duration'] for e in events]
            timestamps = [e['timestamp'] for e in events]
            
            analysis[pattern_key] = {
                'event_count': len(events),
                'avg_duration': np.mean(durations) if durations else 0,
                'duration_std': np.std(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'time_span_hours': (max(timestamps) - min(timestamps)) / 3600 if len(timestamps) > 1 else 0,
                'execution_rate_per_hour': len(events) / max(1, (max(timestamps) - min(timestamps)) / 3600) if len(timestamps) > 1 else 0
            }
        
        return analysis
    
    def _analyze_risk_distribution(self, processor: AdvancedTelemetryProcessor) -> Dict:
        """Analyze risk score distribution"""
        return {
            'economic_risk_events': processor.metrics.economic_risk_events,
            'fallback_events': processor.metrics.fallback_events,
            'anomaly_distribution': dict(processor.metrics.anomaly_counts),
            'total_risk_events': processor.metrics.economic_risk_events + processor.metrics.fallback_events
        }
    
    def _generate_performance_insights(self, processor: AdvancedTelemetryProcessor) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if processor.metrics.fallback_events > 0:
            fallback_rate = processor.metrics.fallback_events / processor.metrics.structured_logs
            insights.append(f"Fallback mode activated in {fallback_rate:.1%} of executions")
        
        if processor.metrics.economic_risk_events > 0:
            risk_rate = processor.metrics.economic_risk_events / processor.metrics.structured_logs
            insights.append(f"Economic risk detected in {risk_rate:.1%} of executions")
        
        # Anomaly insights
        if processor.metrics.anomaly_counts:
            most_common_anomaly = max(processor.metrics.anomaly_counts.items(), key=lambda x: x[1])
            insights.append(f"Most common anomaly type: {most_common_anomaly[0]} ({most_common_anomaly[1]} occurrences)")
        
        return insights
    
    def _generate_recommendations(self, processor: AdvancedTelemetryProcessor) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if processor.metrics.fallback_events / processor.metrics.structured_logs > 0.1:
            recommendations.append("High fallback rate detected - investigate function reliability")
        
        if processor.metrics.economic_risk_events > 0:
            recommendations.append("Economic risk events detected - review billing and invocation patterns")
        
        if 'cold_start' in processor.metrics.anomaly_counts and processor.metrics.anomaly_counts['cold_start'] > 5:
            recommendations.append("Frequent cold starts detected - consider provisioned concurrency")
        
        return recommendations

def fetch_cloudwatch_logs() -> Tuple[bool, Dict]:
    """Fetch logs from CloudWatch with enhanced error handling"""
    print("📡 Fetching logs from CloudWatch...")
    
    # Calculate time window
    end_time = int(time.time() * 1000)
    start_time = int((time.time() - CONFIG['lookback_hours'] * 3600) * 1000)
    
    try:
        result = subprocess.run([
            "aws", "logs", "filter-log-events",
            "--log-group-name", CONFIG['log_group'],
            "--limit", str(CONFIG['max_events']),
            "--start-time", str(start_time),
            "--end-time", str(end_time)
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return False, {'error': f"AWS CLI error: {result.stderr.strip()}"}
        
        if not result.stdout.strip():
            return False, {'error': "No output received from AWS CLI"}
        
        try:
            log_data = json.loads(result.stdout)
            return True, log_data
        except json.JSONDecodeError as e:
            return False, {'error': f"JSON decode error: {str(e)}"}
    
    except subprocess.TimeoutExpired:
        return False, {'error': "AWS CLI command timed out"}
    except Exception as e:
        return False, {'error': f"Unexpected error: {str(e)}"}

def process_log_events(events: List[Dict]) -> Tuple[List[Dict], List[str], TelemetryMetrics]:
    """Process raw log events with advanced analytics"""
    processor = AdvancedTelemetryProcessor()
    structured_logs = []
    side_traces = []
    
    for event in events:
        message = event.get("message", "")
        
        # Process side channel traces
        if message.startswith("[SCAFAD_TRACE]"):
            side_traces.append(message)
            processor.metrics.side_traces += 1
        else:
            # Process structured logs
            try:
                log_entry = json.loads(message)
                enhanced_log = processor.process_structured_log(log_entry)
                
                if enhanced_log.get('status') != 'invalid':
                    structured_logs.append(enhanced_log)
                else:
                    processor.metrics.malformed_logs += 1
                    
            except json.JSONDecodeError:
                processor.metrics.malformed_logs += 1
    
    # Update total metrics
    processor.metrics.total_events = len(events)
    processor.metrics.data_quality_score = calculate_data_quality_score(processor.metrics)
    
    return structured_logs, side_traces, processor.metrics

def calculate_data_quality_score(metrics: TelemetryMetrics) -> float:
    """Calculate overall data quality score"""
    if metrics.total_events == 0:
        return 0.0
    
    # Quality factors
    structured_ratio = metrics.structured_logs / metrics.total_events
    malformed_ratio = metrics.malformed_logs / metrics.total_events
    
    # Base score from successful parsing
    base_score = structured_ratio * 100
    
    # Penalty for malformed logs
    penalty = malformed_ratio * 20
    
    # Bonus for side traces (indicates comprehensive logging)
    side_trace_bonus = min(metrics.side_traces / metrics.total_events * 10, 5)
    
    final_score = max(0, min(100, base_score - penalty + side_trace_bonus))
    return round(final_score, 2)

def generate_summary_report(metrics: TelemetryMetrics, processor: AdvancedTelemetryProcessor, 
                          exports: Dict[str, str]) -> None:
    """Generate comprehensive summary report"""
    print("\n📊" + "=" * 70 + "📊")
    print("  SCAFAD Layer 0 - Telemetry Analysis Summary")
    print("📊" + "=" * 70 + "📊")
    
    # Basic metrics
    print(f"\n📈 Collection Summary:")
    print(f"   Total Events: {metrics.total_events}")
    print(f"   Structured Logs: {metrics.structured_logs}")
    print(f"   Side Traces: {metrics.side_traces}")
    print(f"   Malformed Logs: {metrics.malformed_logs}")
    print(f"   Data Quality Score: {metrics.data_quality_score}%")
    
    # Anomaly analysis
    if metrics.anomaly_counts:
        print(f"\n🔍 Anomaly Distribution:")
        for anomaly_type, count in sorted(metrics.anomaly_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / metrics.structured_logs) * 100 if metrics.structured_logs > 0 else 0
            print(f"   • {anomaly_type:<20} → {count:>3} ({percentage:>5.1f}%)")
    
    # Risk metrics
    print(f"\n⚠️ Risk Analysis:")
    print(f"   Fallback Events: {metrics.fallback_events}")
    print(f"   Economic Risk Events: {metrics.economic_risk_events}")
    
    if metrics.structured_logs > 0:
        fallback_rate = (metrics.fallback_events / metrics.structured_logs) * 100
        risk_rate = (metrics.economic_risk_events / metrics.structured_logs) * 100
        print(f"   Fallback Rate: {fallback_rate:.1f}%")
        print(f"   Economic Risk Rate: {risk_rate:.1f}%")
    
    # Export summary
    print(f"\n💾 Export Summary:")
    for export_type, filepath in exports.items():
        filename = os.path.basename(filepath)
        print(f"   • {export_type.upper()}: {filename}")
    
    # Processing time
    if hasattr(metrics, 'processing_time_ms'):
        print(f"\n⏱️ Processing Time: {metrics.processing_time_ms:.2f}ms")
    
    print("\n" + "📊" + "=" * 70 + "📊")

async def main():
    """Main async function for enhanced telemetry processing"""
    start_time = time.time()
    
    print("🚀 SCAFAD Layer 0 - Enhanced Telemetry Harvester")
    print(f"Version: {CONFIG['telemetry_version']}")
    print("=" * 80)
    
    # Initialize components
    exporter = TelemetryExporter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Fetch logs from CloudWatch
    success, log_data = fetch_cloudwatch_logs()
    
    if not success:
        print(f"❌ Failed to fetch logs: {log_data.get('error', 'Unknown error')}")
        print("\n💡 Troubleshooting steps:")
        print("   1. Check AWS credentials: aws sts get-caller-identity")
        print("   2. Verify log group name in CONFIG")
        print("   3. Ensure proper IAM permissions for CloudWatch Logs")
        print("   4. Check AWS region configuration")
        return False
    
    events = log_data.get("events", [])
    print(f"✅ Retrieved {len(events)} events from CloudWatch")
    
    if not events:
        print("⚠️ No events found in the specified time window")
        return True
    
    # Process events with advanced analytics
    structured_logs, side_traces, metrics = process_log_events(events)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000
    metrics.processing_time_ms = processing_time
    
    # Create processor for advanced analytics
    processor = AdvancedTelemetryProcessor()
    processor.metrics = metrics
    
    # Export data in multiple formats
    exports = {}
    
    if structured_logs:
        log_exports = exporter.export_structured_logs(structured_logs, timestamp)
        exports.update(log_exports)
    
    if side_traces:
        trace_export = exporter.export_side_traces(side_traces, timestamp)
        exports['side_traces'] = trace_export
    
    # Generate analytics report
    if CONFIG['enable_analysis'] and structured_logs:
        analytics_export = exporter.export_analytics_report(processor, timestamp)
        exports['analytics'] = analytics_export
    
    # Generate summary report
    generate_summary_report(metrics, processor, exports)
    
    # Health recommendations
    print("\n💡 Recommendations:")
    if metrics.data_quality_score < 80:
        print("   ⚠️ Data quality below 80% - investigate log format issues")
    if metrics.fallback_events > metrics.structured_logs * 0.1:
        print("   ⚠️ High fallback rate - check function reliability")
    if metrics.economic_risk_events > 0:
        print("   ⚠️ Economic risk detected - review billing patterns")
    if not metrics.anomaly_counts:
        print("   ℹ️ No anomalies detected - system operating normally")
    else:
        print("   ✅ Anomaly detection data collected successfully")
    
    print(f"\n🏁 Processing completed in {processing_time:.2f}ms")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    # Check for command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print("SCAFAD Layer 0 - Enhanced Telemetry Harvester")
            print("=" * 50)
            print("Usage: python fetch_logs.py [options]")
            print("\nOptions:")
            print("  --help              Show this help message")
            print("  --config            Show current configuration")
            print("  --validate          Validate AWS setup")
            print("  --hours N           Set lookback hours (default: 1)")
            print("  --max-events N      Set max events (default: 1000)")
            print("  --no-compression    Disable compression")
            print("  --no-analysis       Disable advanced analysis")
            print("  --log-group NAME    Override log group name")
            print("\nExamples:")
            print("  python fetch_logs.py --hours 2")
            print("  python fetch_logs.py --max-events 500 --no-compression")
            print("  python fetch_logs.py --log-group /aws/lambda/my-function")
            sys.exit(0)
        
        if '--config' in sys.argv:
            print("Current Configuration:")
            print("=" * 30)
            for key, value in CONFIG.items():
                print(f"{key}: {value}")
            sys.exit(0)
        
        if '--validate' in sys.argv:
            print("Validating AWS Setup...")
            print("=" * 30)
            
            # Check AWS CLI
            try:
                result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
                print(f"✅ AWS CLI: {result.stdout.strip()}")
            except FileNotFoundError:
                print("❌ AWS CLI not found")
                sys.exit(1)
            
            # Check credentials
            try:
                result = subprocess.run(['aws', 'sts', 'get-caller-identity'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    identity = json.loads(result.stdout)
                    print(f"✅ AWS Identity: {identity.get('Arn', 'Unknown')}")
                else:
                    print("❌ AWS credentials not configured")
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Error checking credentials: {e}")
                sys.exit(1)
            
            # Check log group access
            try:
                result = subprocess.run([
                    'aws', 'logs', 'describe-log-groups',
                    '--log-group-name-prefix', CONFIG['log_group']
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    log_groups = json.loads(result.stdout)
                    if log_groups.get('logGroups'):
                        print(f"✅ Log group accessible: {CONFIG['log_group']}")
                    else:
                        print(f"⚠️ Log group not found: {CONFIG['log_group']}")
                else:
                    print(f"❌ Cannot access log groups: {result.stderr}")
            except Exception as e:
                print(f"❌ Error checking log group: {e}")
            
            print("\nValidation complete!")
            sys.exit(0)
        
        # Parse other arguments
        for i, arg in enumerate(sys.argv):
            if arg == '--hours' and i + 1 < len(sys.argv):
                try:
                    CONFIG['lookback_hours'] = int(sys.argv[i + 1])
                    print(f"Set lookback hours to: {CONFIG['lookback_hours']}")
                except ValueError:
                    print("Error: --hours requires a number")
                    sys.exit(1)
            
            elif arg == '--max-events' and i + 1 < len(sys.argv):
                try:
                    CONFIG['max_events'] = int(sys.argv[i + 1])
                    print(f"Set max events to: {CONFIG['max_events']}")
                except ValueError:
                    print("Error: --max-events requires a number")
                    sys.exit(1)
            
            elif arg == '--log-group' and i + 1 < len(sys.argv):
                CONFIG['log_group'] = sys.argv[i + 1]
                print(f"Set log group to: {CONFIG['log_group']}")
            
            elif arg == '--no-compression':
                CONFIG['enable_compression'] = False
                print("Compression disabled")
            
            elif arg == '--no-analysis':
                CONFIG['enable_analysis'] = False
                print("Advanced analysis disabled")
    
    # Run the main function
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)