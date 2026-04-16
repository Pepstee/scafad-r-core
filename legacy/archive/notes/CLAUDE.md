# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCAFAD-R (Resilient Serverless Context-Aware Fusion Anomaly Detection Framework) is an AWS Lambda-based security research project for detecting behavioral anomalies in serverless environments. It implements a 6-layer defense architecture (L0-L6) with ML-powered anomaly detection and MITRE ATT&CK alignment.

## Essential Commands

### Build and Deployment
```bash
# Build Lambda function with SAM
sam build

# Deploy to AWS (guided setup)
sam deploy --guided

# Build using Makefile
make build
make deploy
```

### Testing and Invocation
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Run specific test types
python -m pytest tests/unit/test_handler.py -v  # Unit tests
make test  # Using Makefile

# Invoke Lambda locally with test payload
sam local invoke SCAFADLayer0Function --event event.json
sam local invoke SCAFADLayer0Function --event simple_event.json
make invoke  # Using Makefile

# Simulate multiple anomaly events
python invoke.py --n 10 --mode test --verbose
python invoke.py --n 20 --mode test --adversarial --economic
make invoke-n  # Run 10 simulations via Makefile

# Standalone SCAFAD testing (no SAM required)
python standalone_scafad_test.py
```

### Monitoring and Telemetry
```bash
# Fetch CloudWatch logs
python telemetry/fetch_logs.py
make logs

# View real-time telemetry
tail -f telemetry/invocation_master_log.jsonl | jq '.'

# Generate analytics summary
make summarise
```

### Cleanup
```bash
make clean  # Remove build artifacts and telemetry data
```

## Architecture Overview

### Core Structure
- **app_main.py** - Main Lambda handler and Layer 0 orchestrator with modular architecture
- **app_telemetry.py** - Multi-channel telemetry management and emission system
- **layer0_core.py** - Enhanced anomaly detection with 26 ML algorithms
- **layer0_*.py** - Specialized Layer 0 components (buffer, health, privacy, contracts, etc.)
- **app_config.py** - Centralized configuration management system

### Layer Architecture (L0-L6)
1. **L0**: Adaptive Telemetry Controller - Signal processing and fallback handling
2. **L1**: Behavioral Intake Zone - Data sanitization and schema validation  
3. **L2**: Multi-Vector Detection Matrix - Parallel anomaly detection engines
4. **L3**: Trust-Weighted Fusion - Event-time fusion with volatility suppression
5. **L4**: Explainability & Decision Trace - Auditable scoring and explanations
6. **L5**: Threat Alignment - MITRE ATT&CK mapping and campaign clustering
7. **L6**: Feedback & Learning - Analyst feedback and model adaptation

### Key Components
- **Graph Analysis**: Uses NetworkX for execution flow graphs and provenance tracking
- **ML Detection**: Isolation Forest, i-GNN, and semantic deviation models using scikit-learn/torch
- **Telemetry Processing**: Comprehensive logging with structured JSON payloads
- **Adversarial Simulation**: Economic abuse and attack scenario generation

## Development Notes

### Dependencies
- Python 3.11+ required
- AWS SAM CLI for local testing and deployment
- Core ML libraries: numpy, networkx, scikit-learn, torch
- AWS libraries: boto3 for cloud services integration

### Testing Modes
- `--mode test`: Development testing with verbose output
- `--mode production`: Production simulation mode
- `--adversarial`: Enable attack simulation scenarios
- `--economic`: Include resource abuse detection tests

### Configuration
SAM template configures environment variables in template.yaml:
- `SCAFAD_VERBOSITY`: Logging level (NORMAL/VERBOSE)
- `SCAFAD_ENABLE_*`: Feature toggles for graph, economic, and provenance analysis
- `SCAFAD_TEMPORAL_WINDOW`: Time window for anomaly correlation (300s default)

### File Structure Notes
- `telemetry/` - Contains all execution logs, payloads, and analysis results
- `tests/unit/` - Unit test suite following pytest conventions
- Multiple backup files (*.backup_*) indicate active development - avoid editing backups
- Various test_*.py files for component-specific validation