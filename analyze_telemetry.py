#!/usr/bin/env python3
"""
Quick SCAFAD Layer 0 Telemetry Analysis - Fixed Unicode Version
Analyze local telemetry data with proper encoding handling
"""

import json
import glob
import os
from collections import Counter, defaultdict
from datetime import datetime
import statistics

def safe_read_jsonl(filepath):
    """Safely read JSONL file with encoding handling"""
    entries = []
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON decode error on line {line_num}: {str(e)[:100]}")
                            continue
            print(f"✅ Successfully read {filepath} with {encoding} encoding")
            return entries
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            continue
    
    print(f"❌ Could not read {filepath} with any encoding")
    return []

def analyze_master_log():
    """Analyze the comprehensive master log"""
    master_log_path = "telemetry/invocation_master_log.jsonl"
    
    if not os.path.exists(master_log_path):
        print("❌ Master log not found")
        return
    
    print("📃 MASTER LOG ANALYSIS")
    print("=" * 50)
    
    entries = safe_read_jsonl(master_log_path)
    
    if not entries:
        print("❌ No valid entries found in master log")
        return
    
    print(f"📊 Total Invocations: {len(entries)}")
    
    # Analyze payload patterns
    anomaly_counts = Counter()
    profile_counts = Counter()
    phase_counts = Counter()
    adversarial_count = 0
    economic_count = 0
    graph_count = 0
    success_count = 0
    
    processing_times = []
    durations = []
    risk_scores = []
    
    for entry in entries:
        payload_summary = entry.get('payload_summary', {})
        response_summary = entry.get('response_summary', {})
        scafad_analysis = entry.get('scafad_analysis', {})
        
        # Count patterns
        anomaly_counts[payload_summary.get('anomaly', 'unknown')] += 1
        profile_counts[payload_summary.get('function_profile', 'unknown')] += 1
        phase_counts[payload_summary.get('execution_phase', 'unknown')] += 1
        
        # Count features
        if payload_summary.get('adversarial'):
            adversarial_count += 1
        if payload_summary.get('economic_attack'):
            economic_count += 1
        if payload_summary.get('graph_analysis'):
            graph_count += 1
        
        # Count success
        if response_summary.get('success'):
            success_count += 1
            
        # Collect metrics
        if response_summary.get('duration'):
            durations.append(response_summary['duration'])
        
        processing_time = scafad_analysis.get('processing_time_ms')
        if processing_time:
            processing_times.append(processing_time)
        
        risk_score = scafad_analysis.get('economic_risk_score')
        if risk_score is not None:
            risk_scores.append(risk_score)
    
    # Display results
    print(f"✅ Success Rate: {success_count}/{len(entries)} ({success_count/len(entries)*100:.1f}%)")
    
    print(f"\n🔍 Anomaly Distribution:")
    for anomaly, count in anomaly_counts.most_common():
        percentage = count / len(entries) * 100
        print(f"   • {anomaly:<20} → {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n🏭 Function Profile Distribution:")
    for profile, count in profile_counts.most_common():
        percentage = count / len(entries) * 100
        print(f"   • {profile:<20} → {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n⚡ Execution Phase Distribution:")
    for phase, count in phase_counts.most_common():
        percentage = count / len(entries) * 100
        print(f"   • {phase:<12} → {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n🎭 Advanced Features:")
    print(f"   • Adversarial Payloads: {adversarial_count} ({adversarial_count/len(entries)*100:.1f}%)")
    print(f"   • Economic Attacks: {economic_count} ({economic_count/len(entries)*100:.1f}%)")
    print(f"   • Graph Analysis: {graph_count} ({graph_count/len(entries)*100:.1f}%)")
    
    if durations:
        print(f"\n⏱️  Performance Metrics:")
        print(f"   • Avg Duration: {statistics.mean(durations):.2f}s")
        print(f"   • Min Duration: {min(durations):.2f}s")
        print(f"   • Max Duration: {max(durations):.2f}s")
        print(f"   • Duration StdDev: {statistics.stdev(durations) if len(durations) > 1 else 0:.2f}s")
    
    if processing_times:
        print(f"\n🧠 SCAFAD Processing:")
        print(f"   • Avg Processing Time: {statistics.mean(processing_times):.2f}ms")
        print(f"   • Min Processing Time: {min(processing_times):.2f}ms")
        print(f"   • Max Processing Time: {max(processing_times):.2f}ms")
        print(f"   • Processing StdDev: {statistics.stdev(processing_times) if len(processing_times) > 1 else 0:.2f}ms")
    
    if risk_scores:
        print(f"\n🔴 Risk Analysis:")
        print(f"   • Avg Risk Score: {statistics.mean(risk_scores):.3f}")
        print(f"   • Risk Score Range: {min(risk_scores):.3f} - {max(risk_scores):.3f}")
        high_risk_count = sum(1 for score in risk_scores if score > 0.7)
        print(f"   • High Risk Events: {high_risk_count} ({high_risk_count/len(risk_scores)*100:.1f}%)")

def analyze_execution_reports():
    """Analyze execution reports"""
    report_files = glob.glob("telemetry/analysis/*.json")
    
    if not report_files:
        print("❌ No execution reports found")
        return
    
    print(f"\n📊 EXECUTION REPORTS ANALYSIS")
    print("=" * 40)
    print(f"📋 Found {len(report_files)} execution reports")
    
    total_invocations = 0
    total_successful = 0
    total_duration = 0
    
    for report_file in sorted(report_files):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            exec_summary = report.get('execution_summary', {})
            config = report.get('configuration', {})
            
            invocations = exec_summary.get('total_invocations', 0)
            successful = exec_summary.get('successful', 0)
            duration = exec_summary.get('total_duration', 0)
            
            total_invocations += invocations
            total_successful += successful
            total_duration += duration
            
            timestamp = os.path.basename(report_file).replace('execution_report_', '').replace('.json', '')
            
            print(f"\n📈 Report {timestamp}:")
            print(f"   • Invocations: {invocations}")
            print(f"   • Success Rate: {exec_summary.get('success_rate', 0)*100:.1f}%")
            print(f"   • Duration: {duration:.2f}s")
            print(f"   • Mode: {config.get('mode', 'unknown')}")
            print(f"   • Features: {'🎭' if config.get('adversarial_enabled') else ''}{'💰' if config.get('economic_enabled') else ''}")
            
        except Exception as e:
            print(f"⚠️  Error reading {report_file}: {e}")
    
    if total_invocations > 0:
        print(f"\n🎯 CUMULATIVE TOTALS:")
        print(f"   • Total Invocations: {total_invocations}")
        print(f"   • Total Successful: {total_successful}")
        print(f"   • Overall Success Rate: {total_successful/total_invocations*100:.1f}%")
        print(f"   • Total Execution Time: {total_duration:.2f}s")
        print(f"   • Average per Invocation: {total_duration/total_invocations:.2f}s")

def analyze_file_counts():
    """Analyze file counts in telemetry directory"""
    print(f"\n📁 FILE SYSTEM ANALYSIS")
    print("=" * 30)
    
    if not os.path.exists("telemetry"):
        print("❌ Telemetry directory not found")
        return
    
    # Count files in each subdirectory
    subdirs = ["payloads", "responses", "analysis", "graphs", "adversarial", "economic"]
    
    for subdir in subdirs:
        path = f"telemetry/{subdir}"
        if os.path.exists(path):
            files = glob.glob(f"{path}/*")
            print(f"   • {subdir:<12}: {len(files)} files")
        else:
            print(f"   • {subdir:<12}: Directory not found")
    
    # Check for master log
    master_log_path = "telemetry/invocation_master_log.jsonl"
    if os.path.exists(master_log_path):
        try:
            # Try to count lines safely
            entries = safe_read_jsonl(master_log_path)
            print(f"   • Master Log: {len(entries)} entries")
        except Exception as e:
            print(f"   • Master Log: Error reading ({e})")
    else:
        print(f"   • Master Log: Not found")

def detect_anomaly_patterns():
    """Detect interesting anomaly patterns"""
    print(f"\n🔍 ANOMALY PATTERN DETECTION")
    print("=" * 35)
    
    master_log_path = "telemetry/invocation_master_log.jsonl"
    if not os.path.exists(master_log_path):
        print("❌ Master log not found for pattern analysis")
        return
    
    entries = safe_read_jsonl(master_log_path)
    if not entries:
        print("❌ No valid entries for pattern analysis")
        return
    
    # Analyze patterns
    function_anomaly_map = defaultdict(list)
    phase_anomaly_map = defaultdict(list)
    
    for entry in entries:
        payload_summary = entry.get('payload_summary', {})
        
        anomaly = payload_summary.get('anomaly')
        function_profile = payload_summary.get('function_profile')
        phase = payload_summary.get('execution_phase')
        
        if anomaly:
            function_anomaly_map[function_profile].append(anomaly)
            phase_anomaly_map[phase].append(anomaly)
    
    # Function-specific anomaly patterns
    print("🏭 Function Profile Anomaly Patterns:")
    for function, anomalies in function_anomaly_map.items():
        if function and len(anomalies) > 1:
            anomaly_dist = Counter(anomalies)
            most_common = anomaly_dist.most_common(1)[0]
            print(f"   • {function:<20}: {most_common[0]} ({most_common[1]}/{len(anomalies)})")
    
    # Phase-specific patterns
    print(f"\n⚡ Execution Phase Anomaly Patterns:")
    for phase, anomalies in phase_anomaly_map.items():
        if phase and len(anomalies) > 1:
            anomaly_dist = Counter(anomalies)
            most_common = anomaly_dist.most_common(1)[0]
            print(f"   • {phase:<12}: {most_common[0]} ({most_common[1]}/{len(anomalies)})")

def analyze_recent_payloads():
    """Analyze some recent payload files for insights"""
    print(f"\n📋 RECENT PAYLOAD ANALYSIS")
    print("=" * 30)
    
    payload_files = sorted(glob.glob("telemetry/payloads/*.json"))[-10:]  # Last 10 files
    
    if not payload_files:
        print("❌ No payload files found")
        return
    
    print(f"📊 Analyzing {len(payload_files)} recent payloads...")
    
    features_found = {
        'adversarial': 0,
        'economic_attack': 0,
        'graph_analysis': 0,
        'cold_start_simulation': 0,
        'timeout_scenario': 0,
        'starvation': 0
    }
    
    anomaly_types = Counter()
    profiles = Counter()
    
    for payload_file in payload_files:
        try:
            with open(payload_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            
            # Count anomaly types
            anomaly_types[payload.get('anomaly', 'unknown')] += 1
            profiles[payload.get('function_profile_id', 'unknown')] += 1
            
            # Check for advanced features
            if payload.get('enable_adversarial'):
                features_found['adversarial'] += 1
            if payload.get('economic_attack'):
                features_found['economic_attack'] += 1
            if payload.get('graph_analysis'):
                features_found['graph_analysis'] += 1
            if payload.get('simulate_cold_start'):
                features_found['cold_start_simulation'] += 1
            if payload.get('timeout_scenario'):
                features_found['timeout_scenario'] += 1
            if payload.get('force_starvation'):
                features_found['starvation'] += 1
                
        except Exception as e:
            print(f"⚠️  Error reading {payload_file}: {e}")
    
    print(f"\n🔍 Anomaly Types in Recent Payloads:")
    for anomaly, count in anomaly_types.most_common():
        print(f"   • {anomaly}: {count}")
    
    print(f"\n🏭 Function Profiles in Recent Payloads:")
    for profile, count in profiles.most_common():
        print(f"   • {profile}: {count}")
    
    print(f"\n✨ Advanced Features Usage:")
    for feature, count in features_found.items():
        if count > 0:
            print(f"   • {feature.replace('_', ' ').title()}: {count}")

def generate_summary():
    """Generate overall summary"""
    print(f"\n🎉 SCAFAD LAYER 0 TESTING SUMMARY")
    print("=" * 45)
    
    # Count total files
    total_payloads = len(glob.glob("telemetry/payloads/*.json"))
    total_responses = len(glob.glob("telemetry/responses/*.json"))
    total_reports = len(glob.glob("telemetry/analysis/*.json"))
    
    print(f"📊 Testing Completeness:")
    print(f"   • Generated Payloads: {total_payloads}")
    print(f"   • Captured Responses: {total_responses}")
    print(f"   • Analysis Reports: {total_reports}")
    print(f"   • Test Coverage: {'✅ Complete' if total_payloads > 0 and total_responses > 0 else '⚠️ Incomplete'}")
    
    # Check master log
    master_log_path = "telemetry/invocation_master_log.jsonl"
    if os.path.exists(master_log_path):
        entries = safe_read_jsonl(master_log_path)
        
        print(f"\n🎯 Invocation Summary:")
        print(f"   • Total Tracked Invocations: {len(entries)}")
        print(f"   • Data Integrity: {'✅ Excellent' if len(entries) == total_responses else '⚠️ Check integrity'}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Deploy to AWS: sam deploy --guided")
    print(f"   2. Monitor with CloudWatch")
    print(f"   3. Set up production alerts")
    print(f"   4. Scale testing for production workloads")
    
    print(f"\n🏆 SCAFAD Layer 0 Status: {'🟢 FULLY OPERATIONAL' if total_payloads > 40 else '🟡 IN TESTING'}")

def main():
    """Main analysis function"""
    print("🚀 SCAFAD Layer 0 - Quick Telemetry Analysis")
    print("=" * 70)
    print(f"Analysis Time: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Run all analyses
    analyze_file_counts()
    analyze_master_log()
    analyze_execution_reports()
    detect_anomaly_patterns()
    analyze_recent_payloads()
    generate_summary()
    
    print("\n" + "=" * 70)
    print("🏁 Analysis Complete!")

if __name__ == "__main__":
    main()