# SCAFAD Reproducible Experiments
================================

This directory contains the complete reproducible experiment framework for SCAFAD (Serverless Context-Aware Fusion Anomaly Detection). All experiments are designed to be fully reproducible across different environments using Docker containerization.

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Run all experiments
docker-compose up scafad-experiments

# Run quick validation only
docker-compose up scafad-validation

# Run specific experiment types
docker-compose up scafad-ignn        # i-GNN experiments only
docker-compose up scafad-baselines   # Baseline comparisons only
```

### Option 2: Docker Build and Run

```bash
# Build the SCAFAD experiments container
docker build -t scafad-experiments .

# Run all experiments
docker run -v $(pwd)/experiments/results:/scafad/experiments/results \
           scafad-experiments python experiments/run_reproducible_experiments.py \
           --experiment-type all --seed 42

# Run quick validation
docker run scafad-experiments python experiments/run_reproducible_experiments.py \
           --experiment-type system_validation --quick-mode
```

### Option 3: Local Python Execution

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r docker/requirements-docker.txt

# Run experiments
python experiments/run_reproducible_experiments.py --help
python experiments/run_reproducible_experiments.py --experiment-type all --seed 42
```

## Experiment Types

### 1. System Validation (`system_validation`)
- Validates all SCAFAD components are working correctly
- Tests basic functionality of i-GNN, graph analysis, trace generation
- Quick smoke test for CI/CD pipelines
- **Duration**: ~2-5 minutes

### 2. i-GNN Core Experiments (`ignn_experiments`)
- Comprehensive evaluation of the i-GNN anomaly detection model
- Tests on multiple dataset sizes and anomaly types
- Measures precision, recall, F1-score, and detection time
- **Duration**: ~15-45 minutes (depending on dataset sizes)

### 3. Baseline Comparison (`baseline_experiments`)
- Compares i-GNN against classical ML baselines
- Includes Isolation Forest, One-Class SVM, LOF, DBSCAN
- Statistical significance testing and cross-validation
- **Duration**: ~10-30 minutes

### 4. Formal Verification (`formal_verification_experiments`)
- LTL (Linear Temporal Logic) property verification
- Tests safety, liveness, and causality properties
- Validates trace completeness and system behavior
- **Duration**: ~5-15 minutes

## Configuration Options

### Command Line Arguments

```bash
python experiments/run_reproducible_experiments.py \
    --experiment-type all \          # Type of experiments
    --seed 42 \                      # Random seed for reproducibility
    --output-dir ./results \         # Output directory
    --quick-mode \                   # Reduced dataset sizes for testing
    --resume-from checkpoint.json    # Resume from previous checkpoint
```

### Environment Variables

```bash
export SCAFAD_ENVIRONMENT=DOCKER        # Environment identifier
export SCAFAD_REPRODUCIBLE=true         # Enable reproducibility mode
export SCAFAD_VERBOSITY=NORMAL          # Logging level (NORMAL/VERBOSE/DEBUG)
export SCAFAD_RANDOM_SEED=42            # Global random seed
export AWS_ACCESS_KEY_ID=...            # For AWS Lambda experiments
export AWS_SECRET_ACCESS_KEY=...        # For AWS Lambda experiments
```

## Output Structure

```
experiments/results/
├── logs/                           # Experiment execution logs
│   ├── system_validation_20240101_120000.log
│   ├── ignn_experiments_20240101_120500.log
│   └── ...
├── datasets/                       # Generated datasets
│   ├── normal_traces_seed42.json
│   ├── anomaly_traces_seed42.json
│   └── ...
├── models/                         # Trained model checkpoints
│   ├── ignn_model_seed42.pth
│   ├── baseline_models_seed42.pkl
│   └── ...
├── reports/                        # Final experiment reports
│   ├── final_report.json          # Machine-readable results
│   ├── final_report.html          # Human-readable results
│   └── experiment_summary.pdf     # Academic paper format
└── checkpoints/                    # Experiment checkpoints
    └── experiment_checkpoint.json
```

## Reproducibility Guarantees

### Deterministic Execution
- Fixed random seeds for Python, NumPy, PyTorch, NetworkX
- Deterministic model initialization and training
- Consistent dataset generation across runs
- Controlled execution order and timing

### Environment Isolation
- Docker containerization with fixed dependency versions
- Python environment with `PYTHONHASHSEED` control
- No external network dependencies (except for AWS experiments)
- Isolated filesystem and process space

### Version Control
- All code, configuration, and dependencies versioned
- Docker image tagged with experiment version
- Results include complete environment metadata
- Reproducible builds with locked dependency versions

## Academic Validation

### Statistical Rigor
- Multiple independent runs with different seeds
- Cross-validation for all ML experiments
- Statistical significance testing (p-values, confidence intervals)
- Effect size calculations (Cohen's d, eta-squared)

### Peer Review Support
- Complete experimental protocol documentation
- Raw data and intermediate results preservation
- Reproducible analysis scripts and notebooks
- Academic paper template generation

### Benchmarking Standards
- Standard dataset splits and evaluation metrics
- Comparison with published baseline results
- Runtime and memory usage profiling
- Scalability analysis across dataset sizes

## Interactive Analysis

### Jupyter Notebooks
```bash
# Start Jupyter Lab with experiment access
docker-compose up scafad-jupyter

# Access at http://localhost:8888
# Pre-loaded with experiment results and analysis tools
```

### TensorBoard Monitoring
```bash
# Start TensorBoard for experiment tracking
docker-compose up scafad-tensorboard

# Access at http://localhost:6006
# Real-time monitoring of experiment progress
```

### Database Query Interface
```bash
# PostgreSQL database with experiment metadata
# Connection: postgresql://scafad:scafad_password@localhost:5432/scafad_experiments

# Example queries:
# SELECT * FROM experiment_summary;
# SELECT * FROM model_comparison;
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce dataset sizes with `--quick-mode`
2. **Docker Build Fails**: Check Docker daemon and disk space
3. **Import Errors**: Ensure all dependencies in requirements files
4. **AWS Experiments Fail**: Verify AWS credentials and permissions
5. **Long Runtime**: Use `--quick-mode` for faster testing

### Debug Mode

```bash
# Enable verbose logging
export SCAFAD_VERBOSITY=DEBUG

# Run single experiment with detailed output
python experiments/run_reproducible_experiments.py \
    --experiment-type system_validation --quick-mode
```

### Log Analysis

```bash
# View experiment logs
tail -f experiments/results/logs/*.log

# Parse structured logs
jq '.' experiments/results/logs/system_validation_*.log

# Monitor container resources
docker stats scafad-experiments
```

## Contributing

### Adding New Experiments

1. Create new experiment module in `experiments/`
2. Follow the existing pattern with proper error handling
3. Add configuration options to `run_reproducible_experiments.py`
4. Update documentation and Docker configuration
5. Test reproducibility across multiple seeds

### Modifying Baselines

1. Update `baselines/classical_detectors.py` with new models
2. Add corresponding evaluation metrics
3. Update comparison scripts and reports
4. Verify statistical significance testing

## Citation

If you use this reproducible experiment framework in your research, please cite:

```bibtex
@inproceedings{scafad2024,
  title={SCAFAD: Serverless Context-Aware Fusion Anomaly Detection with Reproducible Experiments},
  author={[Authors]},
  booktitle={[Conference]},
  year={2024}
}
```

## License

This experimental framework is released under the same license as SCAFAD core.