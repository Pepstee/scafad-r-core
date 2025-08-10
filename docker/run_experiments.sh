#!/bin/bash

# SCAFAD Reproducible Experiments Runner
# =====================================
# This script orchestrates the execution of all SCAFAD experiments
# in a reproducible Docker environment with proper logging and checkpointing.

set -e  # Exit on error

echo "🔬 SCAFAD Reproducible Experiments Runner"
echo "========================================"
echo "Container: $(hostname)"
echo "Python: $(python --version)"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"
echo "Timestamp: $(date)"
echo ""

# Parse command line arguments
EXPERIMENT_TYPE="${1:-all}"
QUICK_MODE="${2:-false}"
SEED="${3:-42}"
OUTPUT_DIR="${4:-/scafad/experiments/results}"

echo "🎯 Experiment Configuration:"
echo "   Type: $EXPERIMENT_TYPE"
echo "   Quick Mode: $QUICK_MODE"
echo "   Random Seed: $SEED"
echo "   Output Directory: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/datasets" 
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/reports"
mkdir -p "$OUTPUT_DIR/checkpoints"

# Set environment variables for reproducibility
export PYTHONHASHSEED="$SEED"
export SCAFAD_RANDOM_SEED="$SEED"
export SCAFAD_REPRODUCIBLE="true"
export SCAFAD_OUTPUT_DIR="$OUTPUT_DIR"

# Function to run experiment with logging
run_experiment() {
    local name="$1"
    local script="$2"
    local args="$3"
    
    echo "🧪 Running experiment: $name"
    echo "   Script: $script"
    echo "   Args: $args"
    
    local log_file="$OUTPUT_DIR/logs/${name}_$(date +%Y%m%d_%H%M%S).log"
    local start_time=$(date +%s)
    
    # Run the experiment with timeout and logging
    timeout 3600 python "$script" $args 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $name completed successfully in ${duration}s"
    else
        echo "❌ $name failed with exit code $exit_code after ${duration}s"
        echo "   Check log: $log_file"
    fi
    
    echo ""
    return $exit_code
}

# Main experiment runner
main() {
    echo "🚀 Starting SCAFAD experiments..."
    local overall_start=$(date +%s)
    
    # System validation first
    echo "🔍 System Validation"
    run_experiment "system_validation" "experiments/validate_system.py" "--docker-mode"
    
    if [ "$EXPERIMENT_TYPE" = "validation" ]; then
        echo "🏁 Validation-only mode completed"
        return 0
    fi
    
    # i-GNN experiments
    if [ "$EXPERIMENT_TYPE" = "all" ] || [ "$EXPERIMENT_TYPE" = "ignn" ]; then
        echo "🧠 i-GNN Core Experiments"
        local ignn_args="--seed $SEED --output-dir $OUTPUT_DIR/ignn"
        if [ "$QUICK_MODE" = "true" ]; then
            ignn_args="$ignn_args --quick-mode"
        fi
        
        run_experiment "ignn_training" "experiments/train_ignn.py" "$ignn_args"
        run_experiment "ignn_evaluation" "experiments/evaluate_ignn.py" "$ignn_args" 
        run_experiment "ignn_ablation" "experiments/ablation_ignn.py" "$ignn_args"
    fi
    
    # Baseline comparisons
    if [ "$EXPERIMENT_TYPE" = "all" ] || [ "$EXPERIMENT_TYPE" = "baselines" ]; then
        echo "📊 Baseline Comparison Experiments" 
        local baseline_args="--seed $SEED --output-dir $OUTPUT_DIR/baselines"
        if [ "$QUICK_MODE" = "true" ]; then
            baseline_args="$baseline_args --quick-mode"
        fi
        
        run_experiment "baseline_training" "experiments/train_baselines.py" "$baseline_args"
        run_experiment "baseline_evaluation" "experiments/evaluate_baselines.py" "$baseline_args"
        run_experiment "ignn_vs_baselines" "evaluation/ignn_vs_baselines.py" "$baseline_args"
    fi
    
    # Graph analysis experiments
    if [ "$EXPERIMENT_TYPE" = "all" ] || [ "$EXPERIMENT_TYPE" = "graph" ]; then
        echo "🕸️ Graph Analysis Experiments"
        local graph_args="--seed $SEED --output-dir $OUTPUT_DIR/graph"
        if [ "$QUICK_MODE" = "true" ]; then
            graph_args="$graph_args --quick-mode"
        fi
        
        run_experiment "graph_analysis" "experiments/graph_analysis.py" "$graph_args"
        run_experiment "graph_scalability" "experiments/graph_scalability.py" "$graph_args"
    fi
    
    # Dataset experiments
    if [ "$EXPERIMENT_TYPE" = "all" ] || [ "$EXPERIMENT_TYPE" = "datasets" ]; then
        echo "📈 Dataset Generation and Validation Experiments"
        local dataset_args="--seed $SEED --output-dir $OUTPUT_DIR/datasets"
        if [ "$QUICK_MODE" = "true" ]; then
            dataset_args="$dataset_args --quick-mode"
        fi
        
        run_experiment "dataset_generation" "experiments/generate_datasets.py" "$dataset_args"
        run_experiment "dataset_validation" "experiments/validate_datasets.py" "$dataset_args"
    fi
    
    # AWS Lambda experiments (if credentials available)
    if [ "$EXPERIMENT_TYPE" = "all" ] || [ "$EXPERIMENT_TYPE" = "aws" ]; then
        if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
            echo "☁️ AWS Lambda Deployment Experiments"
            local aws_args="--seed $SEED --output-dir $OUTPUT_DIR/aws"
            if [ "$QUICK_MODE" = "true" ]; then
                aws_args="$aws_args --quick-mode"
            fi
            
            run_experiment "aws_deployment" "experiments/aws_deployment.py" "$aws_args"
            run_experiment "aws_cold_starts" "experiments/aws_cold_start_analysis.py" "$aws_args"
        else
            echo "⚠️  Skipping AWS experiments (no credentials provided)"
        fi
    fi
    
    # Formal verification experiments
    if [ "$EXPERIMENT_TYPE" = "all" ] || [ "$EXPERIMENT_TYPE" = "verification" ]; then
        echo "✅ Formal Verification Experiments"
        local verify_args="--seed $SEED --output-dir $OUTPUT_DIR/verification"
        if [ "$QUICK_MODE" = "true" ]; then
            verify_args="$verify_args --quick-mode"
        fi
        
        run_experiment "ltl_verification" "experiments/ltl_verification.py" "$verify_args"
        run_experiment "property_testing" "experiments/property_testing.py" "$verify_args"
    fi
    
    # Generate final report
    echo "📋 Generating Final Report"
    run_experiment "final_report" "experiments/generate_final_report.py" "--output-dir $OUTPUT_DIR --all-experiments"
    
    local overall_end=$(date +%s)
    local total_duration=$((overall_end - overall_start))
    
    echo "🎉 All SCAFAD experiments completed!"
    echo "   Total time: ${total_duration}s ($(($total_duration / 60)) minutes)"
    echo "   Results directory: $OUTPUT_DIR"
    echo "   Logs directory: $OUTPUT_DIR/logs"
    echo ""
    echo "📄 View the final report:"
    echo "   cat $OUTPUT_DIR/reports/final_report.html"
    echo "   cat $OUTPUT_DIR/reports/final_report.json"
}

# Print help
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "SCAFAD Reproducible Experiments Runner"
    echo ""
    echo "Usage: $0 [EXPERIMENT_TYPE] [QUICK_MODE] [SEED] [OUTPUT_DIR]"
    echo ""
    echo "Arguments:"
    echo "  EXPERIMENT_TYPE  Type of experiments to run (default: all)"
    echo "                   Options: all, validation, ignn, baselines, graph, datasets, aws, verification"
    echo "  QUICK_MODE       Run in quick mode with reduced dataset sizes (default: false)"
    echo "  SEED            Random seed for reproducibility (default: 42)"  
    echo "  OUTPUT_DIR       Directory for experiment outputs (default: /scafad/experiments/results)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all experiments"
    echo "  $0 validation                # Run only validation"
    echo "  $0 ignn true                 # Run i-GNN experiments in quick mode"  
    echo "  $0 all false 123             # Run all experiments with seed 123"
    echo "  $0 baselines true 42 /tmp    # Run baseline experiments in /tmp"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_ACCESS_KEY_ID        AWS credentials for Lambda experiments"
    echo "  AWS_SECRET_ACCESS_KEY    AWS credentials for Lambda experiments"
    echo "  SCAFAD_VERBOSITY         Logging verbosity (NORMAL, VERBOSE, DEBUG)"
    echo ""
    exit 0
fi

# Run main function
main "$@"