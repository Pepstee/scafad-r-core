# SCAFAD Reproducible Experiments Makefile
# =========================================
# This Makefile provides convenient commands for running reproducible
# SCAFAD experiments using Docker containers.

.PHONY: help build test experiments clean validate quick-test full-test evaluate

# ---------------------------------------------------------------------------
# C-5 Evaluation harness — no Docker required
# ---------------------------------------------------------------------------
EVAL_OUTPUT_DIR ?= ./evaluation/results
EVAL_SEED       ?= 42
SCAFAD_PKG      ?= ./scafad

# Run the full permanent test suite (T-013..T-025) and write a JSON artefact.
# Exit code 0 iff all 419 tests pass.
evaluate:
	@echo "SCAFAD-R Evaluation Harness (C-5)"
	@echo "=================================="
	cd $(SCAFAD_PKG) && python3 evaluate_scafad.py \
		--output-dir ../$(EVAL_OUTPUT_DIR) \
		--seed $(EVAL_SEED)

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
help:
	@echo "SCAFAD Reproducible Experiments"
	@echo "==============================="
	@echo ""
	@echo "Available targets:"
	@echo "  build          - Build SCAFAD Docker container"
	@echo "  test           - Run system validation tests"
	@echo "  validate       - Quick validation of SCAFAD system"
	@echo "  experiments    - Run all reproducible experiments"
	@echo "  quick-test     - Run experiments in quick mode"
	@echo "  full-test      - Run comprehensive experiments"
	@echo "  ignn           - Run i-GNN experiments only"
	@echo "  baselines      - Run baseline comparison experiments"
	@echo "  verification   - Run formal verification experiments"
	@echo "  jupyter        - Start Jupyter Lab for interactive analysis"
	@echo "  tensorboard    - Start TensorBoard for experiment monitoring"
	@echo "  clean          - Clean up containers and volumes"
	@echo "  clean-results  - Clean experiment results"
	@echo "  logs           - Show experiment logs"
	@echo "  status         - Show container status"
	@echo ""
	@echo "Environment variables:"
	@echo "  SEED           - Random seed (default: 42)"
	@echo "  OUTPUT_DIR     - Results directory (default: ./experiments/results)"
	@echo "  AWS_PROFILE    - AWS profile for Lambda experiments"

# Configuration
SEED ?= 42
OUTPUT_DIR ?= ./experiments/results
CONTAINER_NAME = scafad-experiments
IMAGE_NAME = scafad-experiments
DOCKER_TAG = latest

# Build SCAFAD Docker container
build:
	@echo "🔨 Building SCAFAD Docker container..."
	docker build -t $(IMAGE_NAME):$(DOCKER_TAG) .
	@echo "✅ Build completed: $(IMAGE_NAME):$(DOCKER_TAG)"

# Quick system validation
validate: build
	@echo "🔍 Running SCAFAD system validation..."
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name $(CONTAINER_NAME)-validate \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python experiments/run_reproducible_experiments.py \
		--experiment-type system_validation \
		--seed $(SEED) \
		--quick-mode
	@echo "✅ System validation completed"

# Run all reproducible experiments
experiments: build
	@echo "🧪 Running all SCAFAD reproducible experiments..."
	@echo "   Seed: $(SEED)"
	@echo "   Output: $(OUTPUT_DIR)"
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python experiments/run_reproducible_experiments.py \
		--experiment-type all \
		--seed $(SEED)
	@echo "✅ All experiments completed"
	@echo "📄 Results available in: $(OUTPUT_DIR)"

# Quick test mode (reduced datasets)
quick-test: build
	@echo "⚡ Running SCAFAD experiments in quick mode..."
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name $(CONTAINER_NAME)-quick \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python experiments/run_reproducible_experiments.py \
		--experiment-type all \
		--seed $(SEED) \
		--quick-mode
	@echo "✅ Quick experiments completed"

# Comprehensive full test
full-test: build
	@echo "🎯 Running comprehensive SCAFAD experiments..."
	@echo "⚠️  This may take 1-2 hours to complete"
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name $(CONTAINER_NAME)-full \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python experiments/run_reproducible_experiments.py \
		--experiment-type all \
		--seed $(SEED)
	@echo "✅ Full experiments completed"

# Run only i-GNN experiments
ignn: build
	@echo "🧠 Running i-GNN experiments..."
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name $(CONTAINER_NAME)-ignn \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python experiments/run_reproducible_experiments.py \
		--experiment-type ignn_experiments \
		--seed $(SEED)

# Run only baseline comparisons
baselines: build
	@echo "📊 Running baseline comparison experiments..."
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name $(CONTAINER_NAME)-baselines \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python experiments/run_reproducible_experiments.py \
		--experiment-type baseline_experiments \
		--seed $(SEED)

# Run only formal verification
verification: build
	@echo "✅ Running formal verification experiments..."
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name $(CONTAINER_NAME)-verification \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python experiments/run_reproducible_experiments.py \
		--experiment-type formal_verification_experiments \
		--seed $(SEED)

# Start Jupyter Lab for interactive analysis
jupyter: build
	@echo "📓 Starting Jupyter Lab..."
	@echo "🌐 Access at: http://localhost:8888"
	docker run --rm -it \
		-p 8888:8888 \
		-v $(PWD):/scafad \
		--name scafad-jupyter \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

# Start TensorBoard for monitoring
tensorboard: build
	@echo "📊 Starting TensorBoard..."
	@echo "🌐 Access at: http://localhost:6006"
	docker run --rm -it \
		-p 6006:6006 \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		--name scafad-tensorboard \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		tensorboard --logdir /scafad/experiments/results/logs --host 0.0.0.0 --port 6006

# Docker Compose based commands
compose-up:
	@echo "🐳 Starting SCAFAD with Docker Compose..."
	docker-compose up -d

compose-experiments:
	@echo "🧪 Running experiments via Docker Compose..."
	docker-compose up scafad-experiments

compose-validation:
	@echo "🔍 Running validation via Docker Compose..."
	docker-compose up scafad-validation

compose-jupyter:
	@echo "📓 Starting Jupyter via Docker Compose..."
	docker-compose up -d scafad-jupyter
	@echo "🌐 Access at: http://localhost:8888"

compose-down:
	@echo "🛑 Stopping Docker Compose services..."
	docker-compose down

# Utility commands
test: validate

# Show experiment logs
logs:
	@echo "📋 SCAFAD experiment logs:"
	@if [ -d "$(OUTPUT_DIR)/logs" ]; then \
		ls -la $(OUTPUT_DIR)/logs/; \
		echo ""; \
		echo "Latest log entries:"; \
		tail -n 20 $(OUTPUT_DIR)/logs/*.log 2>/dev/null | head -50; \
	else \
		echo "No logs found. Run experiments first."; \
	fi

# Show container status
status:
	@echo "🐳 Docker container status:"
	@docker ps -a --filter "name=scafad" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "💾 Docker images:"
	@docker images --filter "reference=scafad*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Clean up containers and volumes
clean:
	@echo "🧹 Cleaning up SCAFAD containers and volumes..."
	-docker stop $$(docker ps -aq --filter "name=scafad")
	-docker rm $$(docker ps -aq --filter "name=scafad")
	-docker volume prune -f
	@echo "✅ Cleanup completed"

# Clean experiment results
clean-results:
	@echo "🗑️  Cleaning experiment results..."
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		rm -rf $(OUTPUT_DIR)/*; \
		echo "✅ Results cleaned: $(OUTPUT_DIR)"; \
	else \
		echo "No results directory found"; \
	fi

# Deep clean (including images)
clean-all: clean
	@echo "🗑️  Deep cleaning (removing images)..."
	-docker rmi $$(docker images --filter "reference=scafad*" -q)
	-docker system prune -f
	@echo "✅ Deep cleanup completed"

# Development helpers
dev-shell: build
	@echo "🐚 Starting development shell..."
	docker run --rm -it \
		-v $(PWD):/scafad \
		--name scafad-dev \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		/bin/bash

# Run specific Python script in container
run-script: build
	@if [ -z "$(SCRIPT)" ]; then \
		echo "❌ Usage: make run-script SCRIPT=path/to/script.py"; \
		exit 1; \
	fi
	@echo "🐍 Running script: $(SCRIPT)"
	docker run --rm \
		-v $(PWD):/scafad \
		--name scafad-script \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python $(SCRIPT)

# Multi-seed experiments for statistical validation
multi-seed-test: build
	@echo "🎲 Running multi-seed experiments for statistical validation..."
	@for seed in 42 123 456 789 999; do \
		echo ""; \
		echo "🧪 Running with seed: $$seed"; \
		docker run --rm \
			-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
			--name $(CONTAINER_NAME)-seed-$$seed \
			$(IMAGE_NAME):$(DOCKER_TAG) \
			python experiments/run_reproducible_experiments.py \
			--experiment-type all \
			--seed $$seed \
			--quick-mode \
			--output-dir experiments/results/multi-seed/seed-$$seed; \
	done
	@echo "✅ Multi-seed experiments completed"

# AWS Lambda experiments (requires AWS credentials)
aws-experiments: build
	@if [ -z "$(AWS_ACCESS_KEY_ID)" ] || [ -z "$(AWS_SECRET_ACCESS_KEY)" ]; then \
		echo "❌ AWS credentials required: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"; \
		exit 1; \
	fi
	@echo "☁️  Running AWS Lambda experiments..."
	docker run --rm \
		-v $(PWD)/$(OUTPUT_DIR):/scafad/experiments/results \
		-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
		-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		-e AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
		--name $(CONTAINER_NAME)-aws \
		$(IMAGE_NAME):$(DOCKER_TAG) \
		python deploy_scafad_to_aws.py \
		--function-name scafad-test \
		--test-cold-starts --test-concurrency --test-performance

# Generate final report
report:
	@echo "📋 Generating comprehe