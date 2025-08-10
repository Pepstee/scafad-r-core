#!/usr/bin/env python3
"""
SCAFAD i-GNN: Intelligent Graph Neural Network for Serverless Anomaly Detection
==============================================================================

Core innovation of SCAFAD - an intelligent Graph Neural Network specifically
designed for serverless execution graph analysis and anomaly detection.

The i-GNN differs from traditional GNNs by:
1. Temporal execution flow awareness
2. Serverless-specific node features (cold starts, memory spikes, duration)
3. Multi-layer attention for both spatial and temporal relationships
4. Real-time inference capability for live anomaly detection
5. Adaptive learning from execution patterns

Based on research:
- "Graph Neural Networks for Anomaly Detection in Dynamic Systems" (2023)
- "Temporal Graph Networks for Serverless Security" (2024)
- "Attention-based GNN for Runtime Anomaly Detection" (2023)
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

# Try to import PyTorch for GNN implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_AVAILABLE = True
except ImportError:
    # Fallback implementation without PyTorch
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using fallback i-GNN implementation.")

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the serverless execution graph"""
    FUNCTION_INVOCATION = "function_invocation"
    COLD_START = "cold_start"
    WARM_START = "warm_start"
    MEMORY_SPIKE = "memory_spike"
    TIMEOUT = "timeout"
    ERROR = "error"
    NETWORK_IO = "network_io"
    DATABASE_ACCESS = "database_access"


class EdgeType(Enum):
    """Types of edges in the serverless execution graph"""
    TEMPORAL_SEQUENCE = "temporal_sequence"      # A->B in time sequence
    CAUSAL_DEPENDENCY = "causal_dependency"      # A causes B
    RESOURCE_SHARING = "resource_sharing"        # A and B share resources
    DATA_FLOW = "data_flow"                      # Data flows from A to B
    INVOCATION_CHAIN = "invocation_chain"        # A invokes B
    CORRELATION = "correlation"                  # A and B are correlated


@dataclass
class ExecutionNode:
    """Represents a node in the serverless execution graph"""
    node_id: str
    node_type: NodeType
    timestamp: float
    function_id: str
    execution_phase: str
    duration: float
    memory_mb: float
    cpu_utilization: float
    network_io_bytes: int
    is_cold_start: bool
    error_occurred: bool
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        if self.features.size == 0:
            self.features = self.compute_features()
    
    def compute_features(self) -> np.ndarray:
        """Compute feature vector for this node"""
        # 15-dimensional feature vector
        features = np.array([
            self.duration,                           # 0: execution duration
            self.memory_mb,                          # 1: memory usage
            self.cpu_utilization / 100.0,           # 2: CPU utilization (normalized)
            float(self.network_io_bytes) / 1024,    # 3: network I/O (KB)
            float(self.is_cold_start),               # 4: cold start indicator
            float(self.error_occurred),              # 5: error indicator
            self.timestamp % 86400,                  # 6: time of day (seconds)
            float(self.node_type == NodeType.FUNCTION_INVOCATION),  # 7
            float(self.node_type == NodeType.COLD_START),           # 8
            float(self.node_type == NodeType.MEMORY_SPIKE),         # 9
            float(self.node_type == NodeType.TIMEOUT),              # 10
            float(self.node_type == NodeType.ERROR),                # 11
            float(self.node_type == NodeType.NETWORK_IO),           # 12
            float(self.node_type == NodeType.DATABASE_ACCESS),      # 13
            hash(self.function_id) % 1000 / 1000.0  # 14: function hash (normalized)
        ], dtype=np.float32)
        
        return features


@dataclass
class ExecutionEdge:
    """Represents an edge in the serverless execution graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    timestamp: float
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        if self.features.size == 0:
            self.features = self.compute_features()
    
    def compute_features(self) -> np.ndarray:
        """Compute feature vector for this edge"""
        # 8-dimensional edge feature vector
        features = np.array([
            self.weight,                                           # 0: edge weight
            float(self.edge_type == EdgeType.TEMPORAL_SEQUENCE),   # 1
            float(self.edge_type == EdgeType.CAUSAL_DEPENDENCY),   # 2
            float(self.edge_type == EdgeType.RESOURCE_SHARING),    # 3
            float(self.edge_type == EdgeType.DATA_FLOW),           # 4
            float(self.edge_type == EdgeType.INVOCATION_CHAIN),    # 5
            float(self.edge_type == EdgeType.CORRELATION),         # 6
            self.timestamp % 86400 / 86400.0                      # 7: normalized time
        ], dtype=np.float32)
        
        return features


class ServerlessExecutionGraph:
    """Represents the complete serverless execution graph"""
    
    def __init__(self, max_nodes: int = 10000):
        self.nodes: Dict[str, ExecutionNode] = {}
        self.edges: List[ExecutionEdge] = []
        self.max_nodes = max_nodes
        self.node_index: Dict[str, int] = {}
        self.created_at = time.time()
        
    def add_node(self, node: ExecutionNode) -> None:
        """Add a node to the graph"""
        if len(self.nodes) >= self.max_nodes:
            # Remove oldest nodes to maintain size limit
            oldest_nodes = sorted(self.nodes.keys(), 
                                key=lambda k: self.nodes[k].timestamp)
            for old_id in oldest_nodes[:len(oldest_nodes) // 4]:  # Remove 25%
                self.remove_node(old_id)
        
        self.nodes[node.node_id] = node
        self.node_index[node.node_id] = len(self.node_index)
        
    def add_edge(self, edge: ExecutionEdge) -> None:
        """Add an edge to the graph"""
        if edge.source_id in self.nodes and edge.target_id in self.nodes:
            self.edges.append(edge)
            
    def remove_node(self, node_id: str) -> None:
        """Remove a node and its associated edges"""
        if node_id in self.nodes:
            # Remove associated edges
            self.edges = [e for e in self.edges 
                         if e.source_id != node_id and e.target_id != node_id]
            
            # Remove node
            del self.nodes[node_id]
            if node_id in self.node_index:
                del self.node_index[node_id]
                
    def get_subgraph(self, center_node_id: str, radius: int = 2) -> 'ServerlessExecutionGraph':
        """Extract a subgraph around a center node"""
        if center_node_id not in self.nodes:
            return ServerlessExecutionGraph()
            
        # BFS to find nodes within radius
        visited = set()
        current_layer = {center_node_id}
        
        for _ in range(radius):
            next_layer = set()
            for node_id in current_layer:
                if node_id not in visited:
                    visited.add(node_id)
                    # Find neighbors
                    for edge in self.edges:
                        if edge.source_id == node_id and edge.target_id not in visited:
                            next_layer.add(edge.target_id)
                        elif edge.target_id == node_id and edge.source_id not in visited:
                            next_layer.add(edge.source_id)
            current_layer = next_layer
            
        # Create subgraph
        subgraph = ServerlessExecutionGraph()
        for node_id in visited:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
                
        # Add edges between nodes in subgraph
        for edge in self.edges:
            if edge.source_id in visited and edge.target_id in visited:
                subgraph.add_edge(edge)
                
        return subgraph
    
    def to_torch_geometric(self) -> Optional[Data]:
        """Convert to PyTorch Geometric Data format"""
        if not TORCH_AVAILABLE or not self.nodes:
            return None
            
        # Create node feature matrix
        node_ids = list(self.nodes.keys())
        node_features = np.stack([self.nodes[nid].features for nid in node_ids])
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge index and edge attributes
        edge_indices = []
        edge_attrs = []
        
        for edge in self.edges:
            if edge.source_id in self.node_index and edge.target_id in self.node_index:
                source_idx = node_ids.index(edge.source_id)
                target_idx = node_ids.index(edge.target_id)
                edge_indices.append([source_idx, target_idx])
                edge_attrs.append(edge.features)
        
        if not edge_indices:
            # Create self-loops if no edges
            num_nodes = len(node_ids)
            edge_indices = [[i, i] for i in range(num_nodes)]
            edge_attrs = [np.zeros(8, dtype=np.float32) for _ in range(num_nodes)]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.stack(edge_attrs), dtype=torch.float)
        
        # Create node labels (0 = normal, 1 = anomalous)
        y = torch.zeros(len(node_ids), dtype=torch.long)
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            # Simple heuristic for anomaly labeling (can be improved)
            if (node.error_occurred or 
                node.duration > 10.0 or 
                node.memory_mb > 1000 or
                node.cpu_utilization > 95):
                y[i] = 1
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


if TORCH_AVAILABLE:
    class TemporalAttentionLayer(nn.Module):
        """Temporal attention mechanism for i-GNN"""
        
        def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            
            # Multi-head attention for temporal relationships
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
            # Linear transformation for temporal features
            self.temporal_transform = nn.Linear(input_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
        def forward(self, x: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Apply temporal attention
            attn_output, _ = self.temporal_attention(x, x, x, attn_mask=temporal_mask)
            
            # Transform and normalize
            output = self.temporal_transform(attn_output)
            output = self.layer_norm(output)
            
            return F.relu(output)


    class iGNN(nn.Module):
        """
        Intelligent Graph Neural Network for Serverless Anomaly Detection
        
        Core innovation combining:
        1. Graph Convolutional layers for spatial relationships
        2. Graph Attention layers for important feature focus
        3. Temporal attention for execution sequence patterns
        4. Multi-task learning for various anomaly types
        """
        
        def __init__(self, 
                     node_features: int = 15,
                     edge_features: int = 8,
                     hidden_dim: int = 64,
                     num_gnn_layers: int = 3,
                     num_classes: int = 2,
                     dropout: float = 0.3):
            super().__init__()
            
            self.node_features = node_features
            self.edge_features = edge_features
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            
            # Initial node embedding
            self.node_embedding = nn.Linear(node_features, hidden_dim)
            
            # GNN layers combining GCN and GAT
            self.gnn_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if i % 2 == 0:
                    # Graph Convolutional layers
                    self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
                else:
                    # Graph Attention layers
                    self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout))
            
            # Temporal attention layer
            self.temporal_attention = TemporalAttentionLayer(hidden_dim, hidden_dim)
            
            # Multi-task heads
            self.anomaly_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
            self.anomaly_score = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            # Anomaly type classifier
            self.anomaly_type_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, len(NodeType))
            )
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, data: Data) -> Dict[str, torch.Tensor]:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = getattr(data, 'batch', None)
            
            # Initial node embedding
            x = self.node_embedding(x)
            x = F.relu(x)
            
            # Apply GNN layers
            for i, gnn_layer in enumerate(self.gnn_layers):
                if isinstance(gnn_layer, GCNConv):
                    x = gnn_layer(x, edge_index)
                else:  # GATConv
                    x = gnn_layer(x, edge_index)
                    
                x = F.relu(x)
                x = self.dropout(x)
            
            # Apply temporal attention (reshape for sequence processing)
            if batch is not None:
                # Group by batch for temporal attention
                batch_size = int(batch.max()) + 1
                x_temporal = []
                for b in range(batch_size):
                    batch_mask = (batch == b)
                    batch_x = x[batch_mask].unsqueeze(0)  # Add sequence dimension
                    temporal_out = self.temporal_attention(batch_x)
                    x_temporal.append(temporal_out.squeeze(0))
                x = torch.cat(x_temporal, dim=0)
            else:
                x = self.temporal_attention(x.unsqueeze(0)).squeeze(0)
            
            # Multi-task outputs
            anomaly_logits = self.anomaly_classifier(x)
            anomaly_scores = self.anomaly_score(x)
            anomaly_type_logits = self.anomaly_type_classifier(x)
            
            return {
                'anomaly_logits': anomaly_logits,
                'anomaly_scores': anomaly_scores.squeeze(-1),
                'anomaly_types': anomaly_type_logits
            }
    
    
    class iGNNTrainer:
        """Training and inference manager for i-GNN"""
        
        def __init__(self, model: iGNN, learning_rate: float = 0.001, device: str = 'cpu'):
            self.model = model
            self.device = torch.device(device)
            self.model.to(self.device)
            
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
            
            # Loss functions
            self.classification_loss = nn.CrossEntropyLoss()
            self.regression_loss = nn.MSELoss()
            self.type_loss = nn.CrossEntropyLoss()
            
        def train_step(self, data: Data, labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
            """Single training step"""
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute losses
            anomaly_loss = self.classification_loss(outputs['anomaly_logits'], labels['anomaly_labels'])
            score_loss = self.regression_loss(outputs['anomaly_scores'], labels['anomaly_scores'])
            type_loss = self.type_loss(outputs['anomaly_types'], labels['anomaly_types'])
            
            # Combined loss with weights
            total_loss = 0.5 * anomaly_loss + 0.3 * score_loss + 0.2 * type_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            return {
                'total_loss': total_loss.item(),
                'anomaly_loss': anomaly_loss.item(),
                'score_loss': score_loss.item(),
                'type_loss': type_loss.item()
            }
        
        def predict(self, data: Data) -> Dict[str, np.ndarray]:
            """Inference on new data"""
            self.model.eval()
            
            with torch.no_grad():
                data = data.to(self.device)
                outputs = self.model(data)
                
                # Convert to probabilities
                anomaly_probs = F.softmax(outputs['anomaly_logits'], dim=-1)
                type_probs = F.softmax(outputs['anomaly_types'], dim=-1)
                
                return {
                    'anomaly_probabilities': anomaly_probs.cpu().numpy(),
                    'anomaly_scores': outputs['anomaly_scores'].cpu().numpy(),
                    'anomaly_type_probabilities': type_probs.cpu().numpy(),
                    'predicted_labels': torch.argmax(anomaly_probs, dim=-1).cpu().numpy(),
                    'predicted_types': torch.argmax(type_probs, dim=-1).cpu().numpy()
                }


else:
    # Fallback implementation without PyTorch
    class iGNN:
        """Fallback i-GNN implementation without PyTorch"""
        
        def __init__(self, **kwargs):
            self.node_features = kwargs.get('node_features', 15)
            self.hidden_dim = kwargs.get('hidden_dim', 64)
            self.weights_initialized = False
            logger.warning("Using fallback i-GNN implementation. Install PyTorch for full functionality.")
            
        def predict(self, graph: ServerlessExecutionGraph) -> Dict[str, Any]:
            """Fallback prediction using simple heuristics"""
            if not graph.nodes:
                return {'anomaly_scores': [], 'predicted_labels': []}
            
            scores = []
            labels = []
            
            for node_id, node in graph.nodes.items():
                # Simple anomaly score based on heuristics
                score = 0.0
                
                # Duration anomaly
                if node.duration > 10.0:
                    score += 0.3
                
                # Memory anomaly
                if node.memory_mb > 1000:
                    score += 0.3
                
                # CPU anomaly
                if node.cpu_utilization > 95:
                    score += 0.2
                
                # Error indicator
                if node.error_occurred:
                    score += 0.5
                
                # Cold start penalty for long duration
                if node.is_cold_start and node.duration > 5.0:
                    score += 0.2
                
                score = min(1.0, score)  # Cap at 1.0
                scores.append(score)
                labels.append(1 if score > 0.5 else 0)
            
            return {
                'anomaly_scores': np.array(scores),
                'predicted_labels': np.array(labels),
                'anomaly_probabilities': np.column_stack([1-np.array(scores), scores])
            }
    
    class iGNNTrainer:
        """Fallback trainer"""
        
        def __init__(self, model: iGNN, **kwargs):
            self.model = model
            logger.warning("Using fallback trainer. Install PyTorch for full training capability.")
        
        def train_step(self, graph: ServerlessExecutionGraph, labels: Dict[str, Any]) -> Dict[str, float]:
            """Fallback training (no actual training)"""
            return {'total_loss': 0.0, 'anomaly_loss': 0.0}
        
        def predict(self, graph: ServerlessExecutionGraph) -> Dict[str, Any]:
            """Fallback prediction"""
            return self.model.predict(graph)


class ServerlessGraphBuilder:
    """Builds serverless execution graphs from telemetry data"""
    
    def __init__(self, correlation_window: float = 300.0):
        self.correlation_window = correlation_window  # 5 minutes
        self.graphs: Dict[str, ServerlessExecutionGraph] = {}
        
    def create_node_from_telemetry(self, telemetry_record: Dict[str, Any]) -> ExecutionNode:
        """Create a graph node from telemetry record"""
        node_type = NodeType.FUNCTION_INVOCATION
        
        # Determine node type based on telemetry
        if telemetry_record.get('is_cold_start', False):
            node_type = NodeType.COLD_START
        elif telemetry_record.get('error_occurred', False):
            node_type = NodeType.ERROR
        elif telemetry_record.get('memory_spike_kb', 0) > 500 * 1024:  # 500MB spike
            node_type = NodeType.MEMORY_SPIKE
        elif telemetry_record.get('duration', 0) > 30:  # 30s timeout
            node_type = NodeType.TIMEOUT
        elif telemetry_record.get('network_io_bytes', 0) > 10 * 1024 * 1024:  # 10MB
            node_type = NodeType.NETWORK_IO
        
        return ExecutionNode(
            node_id=telemetry_record.get('event_id', f"node_{time.time()}"),
            node_type=node_type,
            timestamp=telemetry_record.get('timestamp', time.time()),
            function_id=telemetry_record.get('function_id', 'unknown'),
            execution_phase=telemetry_record.get('execution_phase', 'unknown'),
            duration=telemetry_record.get('duration', 0.0),
            memory_mb=telemetry_record.get('memory_spike_kb', 0) / 1024,
            cpu_utilization=telemetry_record.get('cpu_utilization', 0.0),
            network_io_bytes=telemetry_record.get('network_io_bytes', 0),
            is_cold_start=telemetry_record.get('is_cold_start', False),
            error_occurred=telemetry_record.get('error_occurred', False)
        )
    
    def infer_edges(self, graph: ServerlessExecutionGraph) -> None:
        """Infer edges between nodes based on temporal and causal relationships"""
        nodes = list(graph.nodes.values())
        nodes.sort(key=lambda n: n.timestamp)
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Skip if too far apart in time
                if node2.timestamp - node1.timestamp > self.correlation_window:
                    break
                
                edge_weight = 0.0
                edge_types = []
                
                # Temporal sequence (consecutive in time)
                if j == i + 1:
                    edge_types.append(EdgeType.TEMPORAL_SEQUENCE)
                    edge_weight += 0.8
                
                # Same function invocation chain
                if node1.function_id == node2.function_id:
                    edge_types.append(EdgeType.INVOCATION_CHAIN)
                    edge_weight += 0.6
                
                # Causal dependency (error -> next event)
                if node1.error_occurred and not node2.error_occurred:
                    edge_types.append(EdgeType.CAUSAL_DEPENDENCY)
                    edge_weight += 0.7
                
                # Resource sharing (similar resource usage patterns)
                if (abs(node1.memory_mb - node2.memory_mb) < 100 and
                    abs(node1.cpu_utilization - node2.cpu_utilization) < 20):
                    edge_types.append(EdgeType.RESOURCE_SHARING)
                    edge_weight += 0.4
                
                # Data flow (high network I/O followed by processing)
                if (node1.network_io_bytes > 1024 * 1024 and  # 1MB+
                    node2.duration > 1.0 and
                    node2.timestamp - node1.timestamp < 60):  # Within 1 minute
                    edge_types.append(EdgeType.DATA_FLOW)
                    edge_weight += 0.5
                
                # Create edges for significant relationships
                if edge_weight > 0.3:
                    for edge_type in edge_types:
                        edge = ExecutionEdge(
                            source_id=node1.node_id,
                            target_id=node2.node_id,
                            edge_type=edge_type,
                            weight=edge_weight,
                            timestamp=(node1.timestamp + node2.timestamp) / 2
                        )
                        graph.add_edge(edge)
    
    def build_graph(self, telemetry_records: List[Dict[str, Any]], graph_id: str = "default") -> ServerlessExecutionGraph:
        """Build execution graph from telemetry records"""
        graph = ServerlessExecutionGraph()
        
        # Create nodes
        for record in telemetry_records:
            node = self.create_node_from_telemetry(record)
            graph.add_node(node)
        
        # Infer edges
        self.infer_edges(graph)
        
        # Store graph
        self.graphs[graph_id] = graph
        
        return graph
    
    def get_graph(self, graph_id: str) -> Optional[ServerlessExecutionGraph]:
        """Retrieve a stored graph"""
        return self.graphs.get(graph_id)


class iGNNAnomalyDetector:
    """Main i-GNN anomaly detector combining graph building and ML inference"""
    
    def __init__(self, 
                 correlation_window: float = 300.0,
                 model_config: Optional[Dict[str, Any]] = None):
        self.graph_builder = ServerlessGraphBuilder(correlation_window)
        
        # Initialize i-GNN model
        model_config = model_config or {}
        self.model = iGNN(**model_config)
        self.trainer = iGNNTrainer(self.model)
        
        self.detection_threshold = 0.5
        self.min_graph_size = 5  # Minimum nodes for meaningful detection
        
    def detect_anomalies(self, telemetry_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main anomaly detection function"""
        start_time = time.time()
        
        # Build execution graph
        graph = self.graph_builder.build_graph(telemetry_records)
        
        if len(graph.nodes) < self.min_graph_size:
            return {
                'anomaly_detected': False,
                'reason': f'Insufficient graph size: {len(graph.nodes)} < {self.min_graph_size}',
                'processing_time': time.time() - start_time,
                'graph_stats': {
                    'num_nodes': len(graph.nodes),
                    'num_edges': len(graph.edges)
                }
            }
        
        # Convert to appropriate format and predict
        if TORCH_AVAILABLE:
            torch_data = graph.to_torch_geometric()
            if torch_data is None:
                return {'anomaly_detected': False, 'reason': 'Graph conversion failed'}
            
            predictions = self.trainer.predict(torch_data)
        else:
            predictions = self.trainer.predict(graph)
        
        # Analyze predictions
        anomaly_scores = predictions['anomaly_scores']
        predicted_labels = predictions['predicted_labels']
        
        # Overall anomaly decision
        max_anomaly_score = float(np.max(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0
        anomaly_detected = max_anomaly_score > self.detection_threshold
        num_anomalous_nodes = int(np.sum(predicted_labels))
        
        # Identify most anomalous nodes
        anomalous_nodes = []
        if len(anomaly_scores) > 0:
            node_ids = list(graph.nodes.keys())
            top_indices = np.argsort(anomaly_scores)[-min(5, len(anomaly_scores)):][::-1]
            
            for idx in top_indices:
                if idx < len(node_ids) and anomaly_scores[idx] > self.detection_threshold:
                    node = graph.nodes[node_ids[idx]]
                    anomalous_nodes.append({
                        'node_id': node.node_id,
                        'node_type': node.node_type.value,
                        'anomaly_score': float(anomaly_scores[idx]),
                        'function_id': node.function_id,
                        'timestamp': node.timestamp,
                        'duration': node.duration,
                        'memory_mb': node.memory_mb
                    })
        
        processing_time = time.time() - start_time
        
        return {
            'anomaly_detected': anomaly_detected,
            'max_anomaly_score': max_anomaly_score,
            'num_anomalous_nodes': num_anomalous_nodes,
            'total_nodes_analyzed': len(graph.nodes),
            'anomalous_nodes': anomalous_nodes,
            'graph_stats': {
                'num_nodes': len(graph.nodes),
                'num_edges': len(graph.edges),
                'node_types': {nt.value: sum(1 for n in graph.nodes.values() if n.node_type == nt) 
                              for nt in NodeType}
            },
            'processing_time': processing_time,
            'model_type': 'PyTorch i-GNN' if TORCH_AVAILABLE else 'Fallback Heuristic'
        }
    
    def train_on_data(self, training_data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]) -> Dict[str, Any]:
        """Train the i-GNN on labeled data"""
        if not TORCH_AVAILABLE:
            return {'status': 'training_skipped', 'reason': 'PyTorch not available'}
        
        training_losses = []
        
        for telemetry_records, labels in training_data:
            graph = self.graph_builder.build_graph(telemetry_records)
            torch_data = graph.to_torch_geometric()
            
            if torch_data is not None:
                # Convert labels to torch format
                torch_labels = {
                    'anomaly_labels': torch.tensor(labels.get('anomaly_labels', []), dtype=torch.long),
                    'anomaly_scores': torch.tensor(labels.get('anomaly_scores', []), dtype=torch.float),
                    'anomaly_types': torch.tensor(labels.get('anomaly_types', []), dtype=torch.long)
                }
                
                losses = self.trainer.train_step(torch_data, torch_labels)
                training_losses.append(losses)
        
        if training_losses:
            avg_losses = {
                key: sum(loss[key] for loss in training_losses) / len(training_losses)
                for key in training_losses[0].keys()
            }
            return {'status': 'training_completed', 'average_losses': avg_losses}
        else:
            return {'status': 'no_training_data'}
    
    async def run_self_test(self) -> Dict[str, Any]:
        """Self-test of the i-GNN system"""
        try:
            # Generate test telemetry data
            test_records = [
                {
                    'event_id': f'test_{i}',
                    'timestamp': time.time() + i,
                    'function_id': 'test_function',
                    'execution_phase': 'INVOKE',
                    'duration': 1.0 + i * 0.5,
                    'memory_spike_kb': 1024 * (i + 1),
                    'cpu_utilization': 50.0 + i * 10,
                    'network_io_bytes': 1000 * i,
                    'is_cold_start': i == 0,
                    'error_occurred': i == 4  # Last record has error
                }
                for i in range(6)
            ]
            
            # Test anomaly detection
            result = self.detect_anomalies(test_records)
            
            success_indicators = [
                result.get('total_nodes_analyzed', 0) >= 5,
                result.get('processing_time', float('inf')) < 5.0,
                'graph_stats' in result,
                'anomaly_detected' in result
            ]
            
            return {
                'self_test_passed': all(success_indicators),
                'indicators': {
                    'nodes_analyzed': result.get('total_nodes_analyzed', 0),
                    'processing_time': result.get('processing_time', 0),
                    'anomaly_detected': result.get('anomaly_detected', False),
                    'model_type': result.get('model_type', 'unknown')
                },
                'test_result': result
            }
            
        except Exception as e:
            return {
                'self_test_passed': False,
                'error': str(e),
                'model_available': TORCH_AVAILABLE
            }