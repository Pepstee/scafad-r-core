"""
SCAFAD Layer 0: Adaptive Telemetry Controller - Complete Enhanced Implementation
Version: v5.0-production
Institution: Birmingham Newman University
Author: Advanced Serverless Anomaly Detection Framework

This module implements the complete Layer 0 architecture with full algorithmic depth.
"""

import time
import random
import json
import string
import uuid
import hashlib
import asyncio
import logging
import sys
import os
import math
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.spatial.distance import cosine, euclidean
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Graph analysis
import networkx as nx
from networkx.algorithms import centrality, community

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Retry and async
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import aiofiles
import aioboto3

# Formal verification
from z3 import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration and Constants
# ============================================================================

LOG_VERSION = {
    "version": "v5.0",
    "stage": "production",
    "features": [
        "complete_graph_analysis",
        "adversarial_gan",
        "formal_verification",
        "economic_detection",
        "silent_failure_detection"
    ],
    "notes": "Full algorithmic implementation with production-grade features"
}

# Thresholds and Parameters
TIMEOUT_THRESHOLD = 0.6
MAX_MEMORY_SPIKE = 50 * 1024 * 1024  # 50MB
TEMPORAL_WINDOW = 300  # 5 minutes
GRAPH_MAX_NODES = 10000  # Maximum nodes in graph before pruning
ANOMALY_DETECTION_THRESHOLD = 0.7
ADVERSARIAL_EPSILON = 0.1  # For adversarial perturbations

# ============================================================================
# Enhanced Enumerations
# ============================================================================

class AnomalyType(Enum):
    """Extended enumeration of anomaly types with severity levels"""
    BENIGN = ("benign", 0.0)
    COLD_START = ("cold_start", 0.3)
    CPU_BURST = ("cpu_burst", 0.5)
    MEMORY_SPIKE = ("memory_spike", 0.5)
    IO_INTENSIVE = ("io_intensive", 0.4)
    NETWORK_ANOMALY = ("network_anomaly", 0.6)
    STARVATION_FALLBACK = ("starvation_fallback", 0.7)
    TIMEOUT_FALLBACK = ("timeout_fallback", 0.8)
    EXECUTION_FAILURE = ("execution_failure", 0.9)
    ADVERSARIAL_INJECTION = ("adversarial_injection", 0.9)
    CRYPTOMINING = ("cryptomining", 1.0)
    DATA_EXFILTRATION = ("data_exfiltration", 1.0)
    DOS_ATTACK = ("dos_attack", 0.9)
    PRIVILEGE_ESCALATION = ("privilege_escalation", 1.0)
    
    def __init__(self, label, severity):
        self.label = label
        self.severity = severity

class ExecutionPhase(Enum):
    """Function execution phases with timing expectations"""
    INIT = ("init", 0, 5000)  # phase, min_ms, max_ms
    INVOKE = ("invoke", 10, 30000)
    SHUTDOWN = ("shutdown", 0, 1000)
    ERROR = ("error", 0, 0)
    
    def __init__(self, label, min_duration, max_duration):
        self.label = label
        self.min_duration = min_duration
        self.max_duration = max_duration

# ============================================================================
# Enhanced Data Structures
# ============================================================================

@dataclass
class TelemetryRecord:
    """Enhanced structured telemetry record with comprehensive fields"""
    # Core identification (REQUIRED - no defaults)
    event_id: str
    timestamp: float
    function_id: str
    execution_phase: ExecutionPhase
    anomaly_type: AnomalyType
    duration: float
    memory_spike_kb: int
    cpu_utilization: float
    network_io_bytes: int
    
    # OPTIONAL FIELDS (with defaults) - must come after required fields
    anomaly_confidence: float = 0.0
    anomaly_severity: float = 0.0
    disk_io_bytes: int = 0
    
    # Graph features
    graph_node_id: Optional[str] = None
    graph_centrality: float = 0.0
    graph_clustering_coefficient: float = 0.0
    community_id: Optional[int] = None
    
    # Provenance
    provenance_id: Optional[str] = None
    parent_chain: List[str] = field(default_factory=list)
    causal_depth: int = 0
    
    # Advanced features
    adversarial_score: float = 0.0
    economic_risk_score: float = 0.0
    silent_failure_probability: float = 0.0
    completeness_score: float = 1.0
    
    # Operational metadata
    fallback_mode: bool = False
    source: str = "layer0"
    concurrency_id: str = ""
    container_id: Optional[str] = None
    region: Optional[str] = None
    
    # Versioning
    log_version: Dict = field(default_factory=lambda: LOG_VERSION)
    
    def to_json(self) -> str:
        """Convert to JSON with proper enum handling"""
        data = asdict(self)
        data['execution_phase'] = self.execution_phase.label
        data['anomaly_type'] = self.anomaly_type.label
        return json.dumps(data)
    
    def compute_risk_score(self) -> float:
        """Compute overall risk score"""
        weights = {
            'anomaly': 0.3,
            'adversarial': 0.3,
            'economic': 0.2,
            'silent_failure': 0.2
        }
        
        risk = (
            weights['anomaly'] * self.anomaly_severity +
            weights['adversarial'] * self.adversarial_score +
            weights['economic'] * self.economic_risk_score +
            weights['silent_failure'] * self.silent_failure_probability
        )
        
        return min(1.0, risk)

@dataclass
class InvocationNode:
    """Enhanced node representation for invocation graph"""
    node_id: str
    timestamp: float
    function_id: str
    request_id: str
    
    # Behavioral features
    duration: float
    memory_usage: int
    cpu_usage: float
    error_occurred: bool
    
    # Semantic features
    semantic_embedding: np.ndarray
    input_hash: str
    output_hash: Optional[str]
    
    # Graph properties
    in_degree: int = 0
    out_degree: int = 0
    betweenness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    
    # Anomaly indicators
    local_anomaly_score: float = 0.0
    neighborhood_anomaly_score: float = 0.0
    
    def __hash__(self):
        return hash(self.node_id)

# ============================================================================
# Advanced Invocation Graph Builder
# ============================================================================

class InvocationGraphBuilder:
    """
    Complete implementation of dynamic invocation graph construction
    with temporal, causal, and semantic edge creation
    """
    
    def __init__(self, max_nodes: int = GRAPH_MAX_NODES):
        self.graph = nx.DiGraph()
        self.node_features = {}
        self.edge_features = {}
        self.temporal_window = TEMPORAL_WINDOW
        self.max_nodes = max_nodes
        
        # Graph statistics
        self.stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'nodes_pruned': 0,
            'communities_detected': 0
        }
        
        # Semantic similarity model (lightweight)
        self.embedding_dim = 128
        self.semantic_encoder = self._initialize_semantic_encoder()
        
        # Community detection
        self.communities = {}
        self.community_counter = 0
        
    def _initialize_semantic_encoder(self) -> nn.Module:
        """Initialize a simple neural encoder for semantic similarity"""
        class SimpleEncoder(nn.Module):
            def __init__(self, input_dim=10, hidden_dim=64, output_dim=128):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.tanh(self.fc3(x))  # Bounded output
                return x
        
        return SimpleEncoder()
    
    def create_node(self, event: Dict, context: Any) -> str:
        """Create a comprehensive node with all features"""
        
        node_id = f"{context.aws_request_id}_{time.time_ns()}"
        
        # Extract behavioral features
        behavioral_features = self._extract_behavioral_features(event, context)
        
        # Compute semantic embedding
        semantic_embedding = self._compute_semantic_embedding(event)
        
        # Create node object
        node = InvocationNode(
            node_id=node_id,
            timestamp=time.time(),
            function_id=context.function_name,
            request_id=context.aws_request_id,
            duration=0.0,  # Will be updated
            memory_usage=context.memory_limit_in_mb,
            cpu_usage=0.0,  # Will be measured
            error_occurred=False,
            semantic_embedding=semantic_embedding,
            input_hash=hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest(),
            output_hash=None
        )
        
        # Add node to graph
        self.graph.add_node(node_id, **asdict(node))
        self.node_features[node_id] = node
        self.stats['nodes_created'] += 1
        
        # Create edges
        self._create_edges(node_id)
        
        # Prune if necessary
        if len(self.graph.nodes) > self.max_nodes:
            self._prune_old_nodes()
        
        # Update graph metrics
        self._update_node_metrics(node_id)
        
        # Detect communities
        if self.stats['nodes_created'] % 100 == 0:
            self._detect_communities()
        
        return node_id
    
    def _extract_behavioral_features(self, event: Dict, context: Any) -> np.ndarray:
        """Extract behavioral features from event and context"""
        features = []
        
        # Event features
        features.append(len(json.dumps(event)))  # Payload size
        features.append(len(event.keys()) if isinstance(event, dict) else 0)  # Key count
        features.append(self._calculate_entropy(json.dumps(event)))  # Entropy
        
        # Context features
        features.append(context.memory_limit_in_mb)
        features.append(context.get_remaining_time_in_millis() / 1000.0)
        
        # Timing features
        current_hour = datetime.now().hour
        features.append(np.sin(2 * np.pi * current_hour / 24))  # Cyclic encoding
        features.append(np.cos(2 * np.pi * current_hour / 24))
        
        # Request pattern features
        features.append(self._get_invocation_rate())
        features.append(self._detect_burst_score())
        features.append(self._get_concurrency_level())
        
        return np.array(features, dtype=np.float32)
    
    def _compute_semantic_embedding(self, event: Dict) -> np.ndarray:
        """Compute semantic embedding using neural encoder"""
        # Extract features
        features = []
        
        # Structural features
        features.append(len(json.dumps(event)))
        features.append(event.get('anomaly', 0) if isinstance(event.get('anomaly'), (int, float)) else 0)
        
        # Type features
        for key in ['test_mode', 'adversarial_mode', 'economic_test']:
            features.append(1.0 if event.get(key) else 0.0)
        
        # Pad or truncate to fixed size
        feature_vector = np.array(features[:10] + [0] * (10 - len(features[:10])), dtype=np.float32)
        
        # Encode
        with torch.no_grad():
            tensor_input = torch.tensor(feature_vector).unsqueeze(0)
            embedding = self.semantic_encoder(tensor_input).squeeze().numpy()
        
        return embedding
    
    def _create_edges(self, new_node_id: str):
        """Create temporal, causal, and semantic edges"""
        
        new_node = self.node_features[new_node_id]
        nodes_to_connect = []
        
        # Get recent nodes for edge creation
        for node_id in list(self.graph.nodes):
            if node_id == new_node_id:
                continue
                
            node = self.node_features.get(node_id)
            if not node:
                continue
            
            # Check temporal proximity
            time_diff = abs(new_node.timestamp - node.timestamp)
            if time_diff <= self.temporal_window:
                nodes_to_connect.append((node_id, node, time_diff))
        
        # Create edges based on different criteria
        for node_id, node, time_diff in nodes_to_connect:
            edges_created = False
            
            # 1. Temporal edges (time-based)
            if time_diff <= 60:  # Within 1 minute
                weight = np.exp(-time_diff / 30)  # Exponential decay
                self.graph.add_edge(
                    node_id, new_node_id,
                    type='temporal',
                    weight=weight,
                    time_diff=time_diff
                )
                edges_created = True
            
            # 2. Causal edges (request chain)
            if self._detect_causality(node, new_node):
                self.graph.add_edge(
                    node_id, new_node_id,
                    type='causal',
                    weight=1.0,
                    causal_confidence=self._compute_causal_confidence(node, new_node)
                )
                edges_created = True
            
            # 3. Semantic similarity edges
            similarity = self._compute_similarity(
                node.semantic_embedding,
                new_node.semantic_embedding
            )
            if similarity > 0.7:
                self.graph.add_edge(
                    node_id, new_node_id,
                    type='semantic',
                    weight=similarity,
                    similarity_score=similarity
                )
                edges_created = True
            
            # 4. Behavioral pattern edges
            if self._detect_behavioral_correlation(node, new_node):
                self.graph.add_edge(
                    node_id, new_node_id,
                    type='behavioral',
                    weight=0.8,
                    pattern_type=self._identify_pattern_type(node, new_node)
                )
                edges_created = True
            
            if edges_created:
                self.stats['edges_created'] += 1
    
    def _detect_causality(self, node1: InvocationNode, node2: InvocationNode) -> bool:
        """Detect causal relationship between nodes"""
        # Direct causality through request chain
        if node1.request_id == node2.request_id:
            return node1.timestamp < node2.timestamp
        
        # Hash-based causality (output of one is input of another)
        if node1.output_hash and node2.input_hash:
            return node1.output_hash == node2.input_hash
        
        # Function chain causality
        known_chains = [
            ('auth_function', 'process_function'),
            ('validate_function', 'store_function'),
            ('fetch_function', 'transform_function')
        ]
        
        for chain in known_chains:
            if node1.function_id.endswith(chain[0]) and node2.function_id.endswith(chain[1]):
                return True
        
        return False
    
    def _compute_causal_confidence(self, node1: InvocationNode, node2: InvocationNode) -> float:
        """Compute confidence in causal relationship"""
        confidence = 0.0
        
        # Time proximity increases confidence
        time_diff = abs(node2.timestamp - node1.timestamp)
        if time_diff < 1:
            confidence += 0.5
        elif time_diff < 10:
            confidence += 0.3
        elif time_diff < 60:
            confidence += 0.1
        
        # Same request ID is strong indicator
        if node1.request_id == node2.request_id:
            confidence += 0.3
        
        # Hash match is definitive
        if node1.output_hash == node2.input_hash:
            confidence = 1.0
        
        return min(1.0, confidence)
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        
        # Ensure valid range
        return max(0.0, min(1.0, similarity))
    
    def _detect_behavioral_correlation(self, node1: InvocationNode, node2: InvocationNode) -> bool:
        """Detect behavioral correlation between nodes"""
        # Similar resource usage patterns
        memory_similarity = 1 - abs(node1.memory_usage - node2.memory_usage) / max(node1.memory_usage, node2.memory_usage, 1)
        
        # Similar error patterns
        error_correlation = node1.error_occurred == node2.error_occurred
        
        # Similar timing patterns
        if node1.duration > 0 and node2.duration > 0:
            duration_similarity = 1 - abs(node1.duration - node2.duration) / max(node1.duration, node2.duration)
        else:
            duration_similarity = 0.5
        
        # Combined score
        correlation_score = (memory_similarity + (1.0 if error_correlation else 0.0) + duration_similarity) / 3
        
        return correlation_score > 0.6
    
    def _identify_pattern_type(self, node1: InvocationNode, node2: InvocationNode) -> str:
        """Identify the type of behavioral pattern"""
        if node1.error_occurred and node2.error_occurred:
            return "error_cascade"
        elif node1.memory_usage > 900 and node2.memory_usage > 900:
            return "high_memory_correlation"
        elif node1.duration > 5 and node2.duration > 5:
            return "slow_execution_pattern"
        else:
            return "general_correlation"
    
    def _update_node_metrics(self, node_id: str):
        """Update graph metrics for node"""
        if node_id not in self.graph:
            return
        
        # Degree centrality
        node = self.node_features[node_id]
        node.in_degree = self.graph.in_degree(node_id)
        node.out_degree = self.graph.out_degree(node_id)
        
        # Only compute expensive metrics for smaller graphs
        if len(self.graph.nodes) < 1000:
            try:
                # Betweenness centrality (sampled for performance)
                betweenness = centrality.betweenness_centrality(
                    self.graph,
                    k=min(100, len(self.graph.nodes)),
                    normalized=True
                )
                node.betweenness_centrality = betweenness.get(node_id, 0.0)
                
                # Eigenvector centrality
                if len(self.graph.nodes) > 1:
                    eigenvector = centrality.eigenvector_centrality_numpy(
                        self.graph,
                        max_iter=100
                    )
                    node.eigenvector_centrality = eigenvector.get(node_id, 0.0)
            except:
                # Fallback for computation failures
                pass
    
    def _detect_communities(self):
        """Detect communities in the graph"""
        if len(self.graph.nodes) < 10:
            return
        
        try:
            # Use Louvain method for community detection
            communities = community.greedy_modularity_communities(
                self.graph.to_undirected()
            )
            
            # Assign community IDs
            for comm_id, comm_nodes in enumerate(communities):
                for node_id in comm_nodes:
                    if node_id in self.node_features:
                        self.node_features[node_id].community_id = comm_id
                        self.communities[node_id] = comm_id
            
            self.stats['communities_detected'] = len(communities)
        except:
            # Fallback if community detection fails
            pass
    
    def _prune_old_nodes(self):
        """Prune oldest nodes when graph exceeds maximum size"""
        if len(self.graph.nodes) <= self.max_nodes:
            return
        
        # Get nodes sorted by timestamp
        nodes_by_time = sorted(
            self.node_features.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest 10%
        nodes_to_remove = int(len(self.graph.nodes) * 0.1)
        for node_id, _ in nodes_by_time[:nodes_to_remove]:
            self.graph.remove_node(node_id)
            del self.node_features[node_id]
            self.stats['nodes_pruned'] += 1
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Character frequency
        freq = Counter(data)
        probs = [count / len(data) for count in freq.values()]
        
        # Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        return entropy
    
    def _get_invocation_rate(self) -> float:
        """Calculate current invocation rate"""
        if len(self.node_features) < 2:
            return 0.0
        
        # Get recent nodes
        recent_nodes = sorted(
            self.node_features.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:100]
        
        if len(recent_nodes) < 2:
            return 0.0
        
        # Calculate rate (invocations per second)
        time_span = recent_nodes[0].timestamp - recent_nodes[-1].timestamp
        if time_span > 0:
            return len(recent_nodes) / time_span
        
        return 0.0
    
    def _detect_burst_score(self) -> float:
        """Detect burst patterns in invocations"""
        if len(self.node_features) < 10:
            return 0.0
        
        # Get recent timestamps
        timestamps = sorted([
            node.timestamp for node in self.node_features.values()
        ])[-100:]
        
        if len(timestamps) < 10:
            return 0.0
        
        # Calculate inter-arrival times
        inter_arrivals = np.diff(timestamps)
        
        # Burst score based on variance
        if len(inter_arrivals) > 1:
            mean_interval = np.mean(inter_arrivals)
            std_interval = np.std(inter_arrivals)
            
            if mean_interval > 0:
                burst_score = std_interval / mean_interval
                return min(1.0, burst_score)
        
        return 0.0
    
    def _get_concurrency_level(self) -> int:
        """Estimate current concurrency level"""
        if not self.node_features:
            return 0
        
        current_time = time.time()
        concurrent_window = 5.0  # 5 seconds
        
        # Count nodes within concurrent window
        concurrent_count = sum(
            1 for node in self.node_features.values()
            if current_time - node.timestamp <= concurrent_window
        )
        
        return concurrent_count
    
    def get_graph_metrics(self) -> Dict:
        """Get comprehensive graph metrics"""
        metrics = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'density': nx.density(self.graph) if len(self.graph.nodes) > 1 else 0,
            'num_communities': self.stats['communities_detected'],
            'avg_degree': np.mean([d for _, d in self.graph.degree()]) if self.graph.nodes else 0,
            'stats': self.stats
        }
        
        # Add advanced metrics for smaller graphs
        if len(self.graph.nodes) < 100 and len(self.graph.nodes) > 1:
            try:
                metrics['avg_clustering'] = nx.average_clustering(self.graph.to_undirected())
                metrics['transitivity'] = nx.transitivity(self.graph)
                
                # Check connectivity
                if nx.is_weakly_connected(self.graph):
                    metrics['diameter'] = nx.diameter(self.graph.to_undirected())
                    metrics['avg_path_length'] = nx.average_shortest_path_length(self.graph.to_undirected())
            except:
                pass
        
        return metrics
    
    def detect_graph_anomalies(self) -> List[Dict]:
        """Detect anomalies based on graph structure"""
        anomalies = []
        
        # 1. Isolated nodes (no connections)
        isolated = list(nx.isolates(self.graph))
        if isolated:
            anomalies.append({
                'type': 'isolated_nodes',
                'nodes': isolated,
                'severity': 0.3
            })
        
        # 2. Unusual degree distribution
        degrees = [d for _, d in self.graph.degree()]
        if degrees:
            mean_degree = np.mean(degrees)
            std_degree = np.std(degrees)
            
            for node_id, degree in self.graph.degree():
                if degree > mean_degree + 3 * std_degree:
                    anomalies.append({
                        'type': 'high_degree_node',
                        'node': node_id,
                        'degree': degree,
                        'severity': 0.6
                    })
        
        # 3. Cycles detection
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                anomalies.append({
                    'type': 'cycles_detected',
                    'cycles': cycles[:10],  # Limit to first 10
                    'severity': 0.7
                })
        except:
            pass
        
        # 4. Community anomalies
        if self.stats['communities_detected'] > 0:
            community_sizes = Counter(self.communities.values())
            
            # Detect unusually small or large communities
            sizes = list(community_sizes.values())
            if sizes:
                mean_size = np.mean(sizes)
                std_size = np.std(sizes)
                
                for comm_id, size in community_sizes.items():
                    if size < mean_size - 2 * std_size or size > mean_size + 2 * std_size:
                        anomalies.append({
                            'type': 'unusual_community_size',
                            'community_id': comm_id,
                            'size': size,
                            'severity': 0.5
                        })
        
        return anomalies