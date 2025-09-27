"""
SCAFAD Layer 0: Adversarial Anomaly Simulation with GAN Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import hashlib

# ============================================================================
# Adversarial GAN Implementation
# ============================================================================

class AnomalyGenerator(nn.Module):
    """Generator network for creating adversarial anomalies"""
    
    def __init__(self, latent_dim: int = 100, feature_dim: int = 20):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Generator architecture
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Hidden layers
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(256, feature_dim),
            nn.Tanh()  # Bounded output [-1, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate anomaly features from latent vector"""
        if condition is not None:
            # Conditional generation
            z = torch.cat([z, condition], dim=1)
        
        return self.model(z)

class AnomalyDiscriminator(nn.Module):
   """Discriminator network for distinguishing real from generated anomalies"""
   
   def __init__(self, feature_dim: int = 20):
       super().__init__()
       
       self.feature_dim = feature_dim
       
       # Discriminator architecture
       self.model = nn.Sequential(
           # Input layer
           nn.Linear(feature_dim, 256),
           nn.LeakyReLU(0.2),
           nn.Dropout(0.3),
           
           # Hidden layers
           nn.Linear(256, 512),
           nn.BatchNorm1d(512),
           nn.LeakyReLU(0.2),
           nn.Dropout(0.3),
           
           nn.Linear(512, 256),
           nn.BatchNorm1d(256),
           nn.LeakyReLU(0.2),
           nn.Dropout(0.3),
           
           nn.Linear(256, 128),
           nn.BatchNorm1d(128),
           nn.LeakyReLU(0.2),
           
           # Output layer
           nn.Linear(128, 1),
           nn.Sigmoid()  # Binary classification
       )
       
       self._initialize_weights()
   
   def _initialize_weights(self):
       """Initialize weights using Xavier initialization"""
       for m in self.modules():
           if isinstance(m, nn.Linear):
               nn.init.xavier_uniform_(m.weight)
               if m.bias is not None:
                   nn.init.zeros_(m.bias)
   
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Classify whether input is real or generated"""
       return self.model(x)

class WassersteinGAN:
   """Wasserstein GAN for stable adversarial anomaly generation"""
   
   def __init__(self, feature_dim: int = 20, latent_dim: int = 100):
       self.feature_dim = feature_dim
       self.latent_dim = latent_dim
       
       # Initialize networks
       self.generator = AnomalyGenerator(latent_dim, feature_dim)
       self.discriminator = AnomalyDiscriminator(feature_dim)
       
       # Optimizers (WGAN uses RMSprop)
       self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=0.00005)
       self.d_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=0.00005)
       
       # Training parameters
       self.clip_value = 0.01  # Weight clipping for WGAN
       self.n_critic = 5  # Train discriminator n_critic times per generator update
       
       # Training history
       self.g_losses = []
       self.d_losses = []
       self.generated_samples = []
   
   def train_step(self, real_data: torch.Tensor) -> Tuple[float, float]:
       """Single training step for WGAN"""
       batch_size = real_data.size(0)
       
       # Train Discriminator
       d_loss_total = 0
       for _ in range(self.n_critic):
           self.d_optimizer.zero_grad()
           
           # Real data
           d_real = self.discriminator(real_data)
           
           # Generated data
           z = torch.randn(batch_size, self.latent_dim)
           fake_data = self.generator(z)
           d_fake = self.discriminator(fake_data.detach())
           
           # Wasserstein loss
           d_loss = -torch.mean(d_real) + torch.mean(d_fake)
           d_loss.backward()
           self.d_optimizer.step()
           
           # Clip discriminator weights
           for p in self.discriminator.parameters():
               p.data.clamp_(-self.clip_value, self.clip_value)
           
           d_loss_total += d_loss.item()
       
       # Train Generator
       self.g_optimizer.zero_grad()
       
       z = torch.randn(batch_size, self.latent_dim)
       fake_data = self.generator(z)
       g_loss = -torch.mean(self.discriminator(fake_data))
       
       g_loss.backward()
       self.g_optimizer.step()
       
       # Store losses
       self.g_losses.append(g_loss.item())
       self.d_losses.append(d_loss_total / self.n_critic)
       
       return g_loss.item(), d_loss_total / self.n_critic
   
   def generate_anomaly(self, n_samples: int = 1, condition: Optional[torch.Tensor] = None) -> np.ndarray:
       """Generate adversarial anomaly samples"""
       self.generator.eval()
       
       with torch.no_grad():
           z = torch.randn(n_samples, self.latent_dim)
           anomalies = self.generator(z, condition)
           
       self.generator.train()
       return anomalies.numpy()

class AdversarialAnomalySimulator:
   """
   Complete adversarial anomaly simulator with multiple attack patterns
   """
   
   def __init__(self):
       # Initialize GAN
       self.gan = WassersteinGAN(feature_dim=20, latent_dim=100)
       
       # Statistical models for different anomaly types
       self.statistical_models = {
           'cold_start': stats.weibull_min(c=2.3, scale=450),
           'memory_spike': stats.betaprime(a=2, b=5, scale=1000),
           'cpu_burst': stats.poisson(mu=3.2),
           'network_latency': stats.lognorm(s=1.2, scale=100),
           'timeout': stats.gamma(a=2, scale=500)
       }
       
       # Attack pattern generators
       self.attack_patterns = {
           'dos_amplification': self._generate_dos_pattern,
           'billing_attack': self._generate_billing_attack,
           'cryptomining': self._generate_mining_pattern,
           'data_exfiltration': self._generate_exfiltration_pattern,
           'privilege_escalation': self._generate_privilege_escalation,
           'timing_attack': self._generate_timing_attack,
           'mimicry': self._generate_mimicry_attack
       }
       
       # Evasion techniques
       self.evasion_strategies = {
           'noise_injection': self._add_gaussian_noise,
           'gradient_masking': self._apply_gradient_masking,
           'input_transformation': self._transform_input,
           'adaptive_perturbation': self._adaptive_perturbation
       }
       
       # Training data buffer
       self.training_buffer = []
       self.max_buffer_size = 10000
       
       # Attack success metrics
       self.attack_metrics = {
           'attempts': 0,
           'successful_evasions': 0,
           'detected_attacks': 0,
           'average_confidence': 0.0
       }
   
   def generate_adversarial_anomaly(self, 
                                   normal_pattern: Dict,
                                   attack_type: str = 'adaptive',
                                   target_model: Optional[any] = None) -> Dict:
       """Generate sophisticated adversarial anomaly"""
       
       self.attack_metrics['attempts'] += 1
       
       if attack_type == 'adaptive':
           # Use GAN for adaptive generation
           anomaly = self._generate_gan_anomaly(normal_pattern, target_model)
       elif attack_type in self.attack_patterns:
           # Use specific attack pattern
           anomaly = self.attack_patterns[attack_type](normal_pattern)
       else:
           # Statistical generation
           anomaly = self._generate_statistical_anomaly(normal_pattern, attack_type)
       
       # Apply evasion techniques
       anomaly = self._apply_evasion(anomaly, target_model)
       
       # Ensure functional correctness
       anomaly = self._ensure_functionality(anomaly, normal_pattern)
       
       # Track metrics
       self._update_metrics(anomaly, target_model)
       
       return anomaly
   
   def _generate_gan_anomaly(self, normal_pattern: Dict, target_model: Optional[any] = None) -> Dict:
       """Generate anomaly using trained GAN"""
       
       # Extract features from normal pattern
       features = self._extract_features(normal_pattern)
       
       # Generate adversarial features
       adversarial_features = self.gan.generate_anomaly(n_samples=1)
       
       # Blend with normal features for subtlety
       blend_ratio = 0.3  # 30% adversarial, 70% normal
       blended_features = (1 - blend_ratio) * features + blend_ratio * adversarial_features[0]
       
       # Convert back to pattern format
       anomaly = self._features_to_pattern(blended_features, normal_pattern)
       
       # Iterative refinement if target model provided
       if target_model:
           anomaly = self._refine_against_model(anomaly, target_model, max_iterations=10)
       
       return anomaly
   
   def _generate_dos_pattern(self, normal_pattern: Dict) -> Dict:
       """Generate Denial of Service attack pattern"""
       anomaly = normal_pattern.copy()
       
       # Amplification characteristics
       anomaly['invocation_rate'] = normal_pattern.get('invocation_rate', 1) * 100
       anomaly['concurrent_requests'] = 1000
       anomaly['payload_size'] = 10 * 1024 * 1024  # 10MB payloads
       anomaly['timeout_frequency'] = 0.3  # 30% timeouts
       
       # Timing pattern (synchronized bursts)
       anomaly['timing_pattern'] = 'synchronized_burst'
       anomaly['burst_interval'] = 0.1  # 100ms between requests
       anomaly['burst_duration'] = 60  # 1 minute bursts
       
       return anomaly
   
   def _generate_billing_attack(self, normal_pattern: Dict) -> Dict:
       """Generate billing/economic attack pattern"""
       anomaly = normal_pattern.copy()
       
       # Maximum resource consumption
       anomaly['memory_usage'] = 3008  # Max Lambda memory
       anomaly['execution_time'] = 899  # Just under 15-minute timeout
       anomaly['invocation_pattern'] = 'periodic'
       anomaly['period'] = 3600  # Hourly to avoid detection
       
       # Subtle patterns to avoid detection
       anomaly['cpu_pattern'] = 'variable'  # Vary CPU to look legitimate
       anomaly['network_io'] = np.random.normal(1000, 200)  # Normal-looking I/O
       
       return anomaly
   
   def _generate_mining_pattern(self, normal_pattern: Dict) -> Dict:
       """Generate cryptomining pattern"""
       anomaly = normal_pattern.copy()
       
       # High, sustained CPU usage
       anomaly['cpu_utilization'] = 95 + np.random.normal(0, 2)
       anomaly['cpu_pattern'] = 'sustained_high'
       anomaly['memory_stable'] = True
       
       # Specific timing to maximize mining
       anomaly['execution_time'] = 840  # 14 minutes (just under timeout)
       anomaly['cold_start_preference'] = False  # Prefer warm containers
       
       # Network pattern for pool communication
       anomaly['network_pattern'] = 'periodic_small'
       anomaly['network_endpoints'] = ['mining_pool_simulated']
       
       return anomaly
   
   def _generate_exfiltration_pattern(self, normal_pattern: Dict) -> Dict:
       """Generate data exfiltration pattern"""
       anomaly = normal_pattern.copy()
       
       # Exfiltration characteristics
       exfil_type = np.random.choice(['slow_leak', 'burst_theft'])
       
       if exfil_type == 'slow_leak':
           anomaly['network_out'] = normal_pattern.get('network_out', 100) * 1.5
           anomaly['network_pattern'] = 'gradual_increase'
           anomaly['payload_chunks'] = 100  # Small chunks
           anomaly['timing'] = 'random_intervals'
       else:  # burst_theft
           anomaly['network_out'] = normal_pattern.get('network_out', 100) * 50
           anomaly['network_pattern'] = 'sudden_spike'
           anomaly['timing'] = 'off_hours'
           anomaly['duration'] = 30  # Quick burst
       
       # Hide in legitimate traffic
       anomaly['mimicry_score'] = 0.8  # 80% similar to normal
       
       return anomaly
   
   def _generate_privilege_escalation(self, normal_pattern: Dict) -> Dict:
       """Generate privilege escalation pattern"""
       anomaly = normal_pattern.copy()
       
       # IAM probing pattern
       anomaly['error_pattern'] = 'permission_denied_sequence'
       anomaly['error_rate'] = 0.4  # 40% errors (probing)
       anomaly['api_calls'] = [
           'iam:ListRoles',
           'iam:GetRole',
           'sts:AssumeRole',
           'lambda:GetFunction',
           'lambda:UpdateFunctionConfiguration'
       ]
       
       # Environment variable access
       anomaly['env_access_pattern'] = 'scanning'
       anomaly['unusual_env_keys'] = [
           'AWS_SECRET_ACCESS_KEY',
           'DATABASE_PASSWORD',
           'API_KEY'
       ]
       
       # Timing to avoid detection
       anomaly['timing_pattern'] = 'distributed'  # Spread over time
       anomaly['delay_between_attempts'] = np.random.uniform(30, 120)
       
       return anomaly
   
   def _generate_timing_attack(self, normal_pattern: Dict) -> Dict:
       """Generate timing-based attack pattern"""
       anomaly = normal_pattern.copy()
       
       # Synchronized timing pattern
       anomaly['execution_pattern'] = 'synchronized_bursts'
       anomaly['sync_interval'] = 1.0  # 1 second intervals
       anomaly['burst_count'] = 10
       anomaly['inter_burst_delay'] = 60  # 1 minute between burst sets
       
       # Measure timing differences
       anomaly['timing_measurements'] = True
       anomaly['response_time_analysis'] = True
       
       return anomaly
   
   def _generate_mimicry_attack(self, normal_pattern: Dict) -> Dict:
       """Generate attack that mimics normal behavior"""
       anomaly = normal_pattern.copy()
       
       # High similarity to normal but with subtle malicious changes
       anomaly['behavioral_similarity'] = 0.95  # 95% similar
       
       # Subtle deviations
       for key in anomaly:
           if isinstance(anomaly[key], (int, float)):
               # Add small perturbation
               anomaly[key] *= np.random.uniform(0.98, 1.02)
       
       # Hidden malicious payload
       anomaly['hidden_command'] = self._encode_command('malicious_action')
       anomaly['steganography'] = True  # Hide data in legitimate fields
       
       return anomaly
   
   def _generate_statistical_anomaly(self, normal_pattern: Dict, anomaly_type: str) -> Dict:
       """Generate anomaly using statistical models"""
       anomaly = normal_pattern.copy()
       
       if anomaly_type in self.statistical_models:
           model = self.statistical_models[anomaly_type]
           
           # Sample from distribution
           if anomaly_type == 'cold_start':
               anomaly['cold_start_duration'] = model.rvs()
               anomaly['init_time'] = anomaly['cold_start_duration'] * 0.8
           elif anomaly_type == 'memory_spike':
               anomaly['memory_spike_mb'] = model.rvs()
               anomaly['memory_pattern'] = 'sudden_spike'
           elif anomaly_type == 'cpu_burst':
               anomaly['cpu_burst_duration'] = model.rvs() * 100
               anomaly['cpu_peak'] = np.random.uniform(80, 100)
           elif anomaly_type == 'network_latency':
               anomaly['network_latency_ms'] = model.rvs()
               anomaly['packet_loss'] = np.random.uniform(0, 0.05)
           elif anomaly_type == 'timeout':
               anomaly['execution_time'] = model.rvs()
               anomaly['timeout_reached'] = anomaly['execution_time'] > 900
       
       return anomaly
   
   def _apply_evasion(self, anomaly: Dict, target_model: Optional[any] = None) -> Dict:
       """Apply evasion techniques to avoid detection"""
       
       # Choose evasion strategy
       strategy = np.random.choice(list(self.evasion_strategies.keys()))
       
       if strategy == 'noise_injection':
           anomaly = self._add_gaussian_noise(anomaly, epsilon=0.1)
       elif strategy == 'gradient_masking':
           anomaly = self._apply_gradient_masking(anomaly, target_model)
       elif strategy == 'input_transformation':
           anomaly = self._transform_input(anomaly)
       elif strategy == 'adaptive_perturbation':
           anomaly = self._adaptive_perturbation(anomaly, target_model)
       
       return anomaly
   
   def _add_gaussian_noise(self, pattern: Dict, epsilon: float = 0.1) -> Dict:
       """Add Gaussian noise to numerical features"""
       noisy_pattern = pattern.copy()
       
       for key, value in pattern.items():
           if isinstance(value, (int, float)):
               noise = np.random.normal(0, epsilon * abs(value))
               noisy_pattern[key] = value + noise
       
       return noisy_pattern
   
   def _apply_gradient_masking(self, pattern: Dict, target_model: Optional[any]) -> Dict:
       """Apply gradient masking to hide from gradient-based detection"""
       masked_pattern = pattern.copy()
       
       if target_model and hasattr(target_model, 'compute_gradient'):
           # Compute gradient and perturb in opposite direction
           gradient = target_model.compute_gradient(pattern)
           
           for key in gradient:
               if key in masked_pattern and isinstance(masked_pattern[key], (int, float)):
                   masked_pattern[key] -= 0.01 * gradient[key]
       
       return masked_pattern
   
   def _transform_input(self, pattern: Dict) -> Dict:
       """Apply input transformations that preserve malicious behavior"""
       transformed = pattern.copy()
       
       # Encoding transformation
       for key in transformed:
           if isinstance(transformed[key], str):
               # Base64 encode strings
               import base64
               transformed[f"{key}_encoded"] = base64.b64encode(
                   transformed[key].encode()
               ).decode()
       
       # Temporal shifting
       if 'timestamp' in transformed:
           transformed['timestamp'] += np.random.uniform(-1, 1)
       
       return transformed
   
   def _adaptive_perturbation(self, pattern: Dict, target_model: Optional[any]) -> Dict:
       """Adaptively perturb based on model feedback"""
       perturbed = pattern.copy()
       
       if not target_model:
           return perturbed
       
       # Binary search for minimum perturbation
       low, high = 0.0, 1.0
       best_perturbation = None
       
       for _ in range(10):  # Binary search iterations
           mid = (low + high) / 2
           test_pattern = self._add_gaussian_noise(pattern, epsilon=mid)
           
           if self._evades_detection(test_pattern, target_model):
               high = mid
               best_perturbation = test_pattern
           else:
               low = mid
       
       return best_perturbation if best_perturbation else perturbed
   
   def _ensure_functionality(self, anomaly: Dict, normal_pattern: Dict) -> Dict:
       """Ensure anomaly maintains functional correctness"""
       
       # Preserve critical fields
       critical_fields = ['function_id', 'request_id', 'region']
       for field in critical_fields:
           if field in normal_pattern:
               anomaly[field] = normal_pattern[field]
       
       # Ensure valid ranges
       if 'cpu_utilization' in anomaly:
           anomaly['cpu_utilization'] = np.clip(anomaly['cpu_utilization'], 0, 100)
       
       if 'memory_usage' in anomaly:
           anomaly['memory_usage'] = np.clip(anomaly['memory_usage'], 0, 3008)
       
       if 'execution_time' in anomaly:
           anomaly['execution_time'] = np.clip(anomaly['execution_time'], 0, 900)
       
       return anomaly
   
   def _refine_against_model(self, anomaly: Dict, target_model: any, max_iterations: int = 10) -> Dict:
       """Iteratively refine anomaly against target detection model"""
       
       refined = anomaly.copy()
       
       for iteration in range(max_iterations):
           if self._evades_detection(refined, target_model):
               # Successfully evades, try to make more subtle
               refined = self._make_more_subtle(refined, anomaly)
           else:
               # Detected, need more evasion
               refined = self._apply_evasion(refined, target_model)
           
           # Check if we've achieved good balance
           if self._is_optimal(refined, target_model):
               break
       
       return refined
   
   def _evades_detection(self, pattern: Dict, model: any) -> bool:
       """Check if pattern evades detection"""
       if not model or not hasattr(model, 'detect'):
           return True
       
       detection_result = model.detect(pattern)
       return detection_result.get('anomaly_score', 0) < 0.5
   
   def _make_more_subtle(self, pattern: Dict, original: Dict) -> Dict:
       """Make pattern more subtle while maintaining effectiveness"""
       subtle = pattern.copy()
       
       # Move values closer to original
       for key in subtle:
           if key in original and isinstance(subtle[key], (int, float)):
               subtle[key] = 0.9 * subtle[key] + 0.1 * original[key]
       
       return subtle
   
   def _is_optimal(self, pattern: Dict, model: any) -> bool:
       """Check if pattern achieves optimal balance"""
       if not model:
           return True
       
       # Should evade detection but still be effective
       evades = self._evades_detection(pattern, model)
       effective = self._is_effective(pattern)
       
       return evades and effective
   
   def _is_effective(self, pattern: Dict) -> bool:
       """Check if attack pattern is still effective"""
       # Check if key attack characteristics are preserved
       if 'dos_amplification' in str(pattern.get('attack_type', '')):
           return pattern.get('invocation_rate', 0) > 50
       elif 'cryptomining' in str(pattern.get('attack_type', '')):
           return pattern.get('cpu_utilization', 0) > 80
       elif 'exfiltration' in str(pattern.get('attack_type', '')):
           return pattern.get('network_out', 0) > 1000
       
       return True
   
   def _extract_features(self, pattern: Dict) -> np.ndarray:
       """Extract numerical features from pattern"""
       features = []
       
       # Standard features
       feature_keys = [
           'cpu_utilization', 'memory_usage', 'execution_time',
           'network_io', 'invocation_rate', 'error_rate',
           'payload_size', 'cold_start_duration'
       ]
       
       for key in feature_keys:
           if key in pattern:
               features.append(float(pattern[key]))
           else:
               features.append(0.0)
       
       # Pad to fixed size
       while len(features) < 20:
           features.append(0.0)
       
       return np.array(features[:20], dtype=np.float32)
   
   def _features_to_pattern(self, features: np.ndarray, template: Dict) -> Dict:
       """Convert feature vector back to pattern dictionary"""
       pattern = template.copy()
       
       feature_keys = [
           'cpu_utilization', 'memory_usage', 'execution_time',
           'network_io', 'invocation_rate', 'error_rate',
           'payload_size', 'cold_start_duration'
       ]
       
       for i, key in enumerate(feature_keys):
           if i < len(features):
               pattern[key] = float(features[i])
       
       return pattern
   
   def _encode_command(self, command: str) -> str:
       """Encode hidden command"""
       import base64
       return base64.b64encode(command.encode()).decode()
   
   def _update_metrics(self, anomaly: Dict, target_model: Optional[any]):
       """Update attack metrics"""
       if target_model:
           if self._evades_detection(anomaly, target_model):
               self.attack_metrics['successful_evasions'] += 1
           else:
               self.attack_metrics['detected_attacks'] += 1
   
   def train_gan(self, real_anomalies: List[Dict], epochs: int = 100):
       """Train GAN on real anomaly data"""
       
       # Convert to feature vectors
       features = np.array([self._extract_features(a) for a in real_anomalies])
       
       # Create dataset
       dataset = torch.utils.data.TensorDataset(
           torch.FloatTensor(features)
       )
       dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
       
       # Training loop
       for epoch in range(epochs):
           epoch_g_loss = 0
           epoch_d_loss = 0
           
           for batch in dataloader:
               real_data = batch[0]
               g_loss, d_loss = self.gan.train_step(real_data)
               
               epoch_g_loss += g_loss
               epoch_d_loss += d_loss
           
           if epoch % 10 == 0:
               logger.info(f"Epoch {epoch}: G_loss={epoch_g_loss:.4f}, D_loss={epoch_d_loss:.4f}")
   
   def inject_runtime_anomaly(self, execution_context: Dict) -> Dict:
       """Inject anomaly into runtime execution"""
       
       anomaly_config = {
           'type': np.random.choice(list(self.attack_patterns.keys())),
           'magnitude': np.random.uniform(0.3, 0.9),
           'duration': np.random.uniform(100, 5000),
           'evasion_enabled': np.random.choice([True, False], p=[0.7, 0.3])
       }
       
       # Generate base anomaly
       if anomaly_config['type'] == 'adaptive':
           anomaly = self._generate_gan_anomaly(execution_context)
       else:
           anomaly = self.attack_patterns[anomaly_config['type']](execution_context)
       
       # Apply evasion if enabled
       if anomaly_config['evasion_enabled']:
           anomaly = self._apply_evasion(anomaly, None)
       
       # Scale by magnitude
       for key in anomaly:
           if isinstance(anomaly[key], (int, float)) and key != 'timestamp':
               anomaly[key] *= anomaly_config['magnitude']
       
       anomaly['injection_config'] = anomaly_config
       
       return anomaly