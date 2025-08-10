-- SCAFAD Experiments Database Schema
-- ===================================
-- This SQL script initializes the PostgreSQL database for tracking
-- SCAFAD experiments, results, and metadata.

-- Create experiment tracking tables
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    seed INTEGER NOT NULL,
    config JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create experiment results table
CREATE TABLE IF NOT EXISTS experiment_results (
    id SERIAL PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(experiment_id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    metadata JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(experiment_id),
    model_name VARCHAR(100) NOT NULL,
    dataset_size INTEGER,
    precision NUMERIC(5,4),
    recall NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    accuracy NUMERIC(5,4),
    auc_score NUMERIC(5,4),
    training_time_seconds NUMERIC(10,3),
    inference_time_ms NUMERIC(10,3),
    memory_usage_mb NUMERIC(10,2),
    parameters JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create dataset tracking
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    dataset_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    size INTEGER NOT NULL,
    num_anomalies INTEGER,
    anomaly_types TEXT[],
    generation_seed INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create experiment-dataset relationships
CREATE TABLE IF NOT EXISTS experiment_datasets (
    experiment_id UUID REFERENCES experiments(experiment_id),
    dataset_id UUID REFERENCES datasets(dataset_id),
    usage_type VARCHAR(20) NOT NULL, -- 'train', 'test', 'validation'
    PRIMARY KEY (experiment_id, dataset_id, usage_type)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(type);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_started_at ON experiments(started_at);
CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment_id ON experiment_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_metric_name ON experiment_results(metric_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_experiment_id ON model_performance(experiment_id);
CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);

-- Insert sample data for testing
INSERT INTO experiments (name, type, status, seed, config) VALUES
    ('system_validation_test', 'validation', 'completed', 42, '{"quick_mode": true}'),
    ('ignn_baseline_comparison', 'comparison', 'running', 123, '{"models": ["ignn", "isolation_forest"]}'),
    ('formal_verification_test', 'verification', 'pending', 456, '{"properties": ["safety", "liveness"]}')
ON CONFLICT (experiment_id) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW experiment_summary AS
SELECT 
    e.experiment_id,
    e.name,
    e.type,
    e.status,
    e.started_at,
    e.completed_at,
    e.duration_seconds,
    COUNT(er.id) as num_metrics,
    COUNT(mp.id) as num_models,
    MAX(mp.f1_score) as best_f1_score
FROM experiments e
LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
LEFT JOIN model_performance mp ON e.experiment_id = mp.experiment_id
GROUP BY e.experiment_id, e.name, e.type, e.status, e.started_at, e.completed_at, e.duration_seconds;

CREATE OR REPLACE VIEW model_comparison AS
SELECT 
    mp.model_name,
    COUNT(mp.id) as num_experiments,
    AVG(mp.f1_score) as avg_f1_score,
    STDDEV(mp.f1_score) as f1_score_std,
    AVG(mp.training_time_seconds) as avg_training_time,
    AVG(mp.inference_time_ms) as avg_inference_time,
    AVG(mp.memory_usage_mb) as avg_memory_usage
FROM model_performance mp
GROUP BY mp.model_name;

-- Create functions for experiment management
CREATE OR REPLACE FUNCTION start_experiment(
    exp_name VARCHAR(100),
    exp_type VARCHAR(50),
    exp_seed INTEGER,
    exp_config JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    exp_id UUID;
BEGIN
    INSERT INTO experiments (name, type, seed, config, status)
    VALUES (exp_name, exp_type, exp_seed, exp_config, 'running')
    RETURNING experiment_id INTO exp_id;
    
    RETURN exp_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION complete_experiment(
    exp_id UUID,
    duration_sec INTEGER DEFAULT NULL,
    error_msg TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE experiments 
    SET 
        status = CASE WHEN error_msg IS NULL THEN 'completed' ELSE 'failed' END,
        completed_at = NOW(),
        duration_seconds = duration_sec,
        error_message = error_msg
    WHERE experiment_id = exp_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_model_performance(
    exp_id UUID,
    model VARCHAR(100),
    ds_size INTEGER,
    prec NUMERIC(5,4),
    rec NUMERIC(5,4),
    f1 NUMERIC(5,4),
    acc NUMERIC(5,4) DEFAULT NULL,
    auc NUMERIC(5,4) DEFAULT NULL,
    train_time NUMERIC(10,3) DEFAULT NULL,
    infer_time NUMERIC(10,3) DEFAULT NULL,
    mem_usage NUMERIC(10,2) DEFAULT NULL,
    params JSONB DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO model_performance (
        experiment_id, model_name, dataset_size,
        precision, recall, f1_score, accuracy, auc_score,
        training_time_seconds, inference_time_ms, memory_usage_mb,
        parameters
    ) VALUES (
        exp_id, model, ds_size, prec, rec, f1, acc, auc,
        train_time, infer_time, mem_usage, params
    );
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions to scafad user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO scafad;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO scafad;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO scafad;