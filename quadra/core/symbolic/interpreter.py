"""
Stateful Symbolic Predictive Interpreter - Complete 8-stage pipeline.

Full inference pipeline:
  1. Encoded Input: Convert input to neural representation
  2. Pattern Extraction: Identify structures (clustering, frequency analysis)
  3. Field Spiking: Generate spike trains via spiking neural networks
  4. Neuroplastic Update: Adapt weights and state based on success
  5. Oscillatory Modulation: Apply temporal phase modulation
  6. Symbolic Interpretation: Derive logical/algebraic insights
  7. Governance Evaluation: Apply policy constraints
  8. Output Synthesis: Format and condition final output

Maintains state across requests via MemoryStore for true adaptive intelligence.
"""

import logging
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from quadra.state.memory_store import MemoryStore, StatefulInferenceContext
from quadra.core.governance.policy_adapter import PolicyEngine, GovernanceService, PolicyContext
from quadra.core.field import FieldEngine
from quadra.core.hrm import SimulationEngine, HRMState

logger = logging.getLogger(__name__)


class InputEncoder:
    """Stage 1: Encode raw input into neural representation."""
    
    def __init__(self, input_dim: int = 128):
        self.input_dim = input_dim
    
    def encode_text(self, text: str) -> np.ndarray:
        """Semantic encoding: weight each token by its WordNet synset richness."""
        clean_text = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        if not clean_text:
            return np.zeros(self.input_dim)

        # Fallback to split-based tokenization if NLTK resources are unavailable.
        try:
            tokens = word_tokenize(clean_text)
        except Exception:
            tokens = clean_text.split()
        
        # Semantic weight = synset count / 15 (normalization constant); 0 for unknown words
        embedding = np.zeros(self.input_dim)
        for i, token in enumerate(tokens[:self.input_dim]):
            token = ''.join(ch for ch in token if ch.isalnum() or ch in ('-', '_'))
            if not token:
                continue
            n_synsets = len(wordnet.synsets(token))
            embedding[i] = n_synsets / 15.0 if n_synsets > 0 else 0.0
        
        # Add tiny noise to avoid zero vectors
        embedding = embedding + np.random.randn(self.input_dim) * 0.0001
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    
    def encode_vector(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector is properly normalized."""
        if vector.size == 0:
            return np.zeros(self.input_dim)
        
        vector = np.array(vector).flatten()
        if vector.size < self.input_dim:
            vector = np.pad(vector, (0, self.input_dim - vector.size), mode='constant')
        elif vector.size > self.input_dim:
            vector = vector[:self.input_dim]
        
        return vector / (np.linalg.norm(vector) + 1e-8)


class PatternExtractor:
    """Stage 2: Extract patterns from encoded input."""
    
    def __init__(self, n_patterns: int = 3):
        self.n_patterns = n_patterns
    
    def extract(self, encoded_input: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extract patterns via frequency analysis and clustering.
        
        Returns:
            patterns: Cluster assignments
            confidence: How confident we are about the patterns
        """
        # Frequency analysis
        fft_result = np.abs(np.fft.fft(encoded_input))
        top_freqs = np.argsort(fft_result)[-self.n_patterns:]
        
        # Simple pattern detection: group by magnitude
        patterns = np.zeros(len(encoded_input))
        for i, idx in enumerate(top_freqs):
            patterns[idx] = i + 1
        
        # Confidence = number of significant semantic peaks / expected richness.
        # With synset-based encoding, each real word creates a peak well above noise.
        abs_vals = np.abs(encoded_input)
        threshold = np.mean(abs_vals) + np.std(abs_vals)
        n_peaks = int(np.sum(abs_vals > threshold))
        confidence = min(1.0, n_peaks / 5.0)
        
        return patterns, confidence


class SpikeGenerator:
    """Stage 3: Generate spike trains via spiking neural networks."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple spiking mechanism
        self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.threshold = 0.5
    
    def forward(self, encoded_input: torch.Tensor, num_steps: int = 5) -> torch.Tensor:
        """
        Generate spike trains by applying threshold over time steps.
        """
        spikes_over_time = []
        membrane_potential = torch.zeros(1, self.hidden_dim)
        
        for step in range(num_steps):
            # Compute input current
            current = torch.matmul(encoded_input.unsqueeze(0), self.weights)
            
            # Update membrane potential
            membrane_potential = 0.9 * membrane_potential + 0.1 * current
            
            # Generate spikes
            spikes = (membrane_potential > self.threshold).float()
            spikes_over_time.append(spikes)
            
            # Reset after spike
            membrane_potential = membrane_potential * (1 - spikes)
        
        return torch.stack(spikes_over_time).mean(dim=0)


class NeuroplasticAdapter:
    """
    Stage 4: Neuroplastic weight adaptation with IFT field modulation.
    
    # Technical implementation
    The synapses sculpt themselves in response to success and failure,
    guided by the field substrate's memory wisdom. When predictions align,
    connections strengthen at rates determined by φ₆'s plasticity signal.
    The adaptation itself becomes plastic—meta-learning through cosmic
    field coupling.
    
    # Implementation
    Implements exponential learning rate adaptation:
        lr(n) = base_lr * (growth_factor)^n * field_factor
    where n = success_streak, growth_factor = 1.1, base_lr = 0.01
    field_factor = 1/(1 + exp(-<φ₆>)) ∈ [0.5, 1.5] (plasticity field)
    
    Properties:
    - Success: streak increments, lr multiplies by 1.1
    - Failure: streak resets to 0, lr returns to base
    - Maximum: capped at 10x speedup (lr_max = 0.1)
    - Doubling time: ~7 consecutive successes (1.1^7 ≈ 1.95)
    - Field modulation: φ₆ scales effective plasticity
    
    Memory mutation: Updates success_streak and learning_rate_trajectory
    """
    
    def __init__(self, memory: MemoryStore, oscillator: Optional['OscillatorModule'] = None):
        self.memory = memory
        self.base_learning_rate = 0.01
        self.oscillator = oscillator  # Reference to get field state
    
    def adapt(self, success: bool) -> np.ndarray:
        """
        Compute adaptive learning rate with field modulation and update memory.
        
        # Technical implementation
        # Implementation
        """
        # # Technical implementation
        # # Implementation
        metrics = self.memory.get_neuroplastic_metrics()
        current_streak = metrics['success_streak']
        
        # # Technical implementation
        # # Implementation
        field_factor = 1.0
        if (self.oscillator is not None and 
            hasattr(self.oscillator, 'enable_ift') and
            self.oscillator.enable_ift and 
            self.oscillator.field_engine is not None):
            phi_6_mean = float(np.mean(self.oscillator.field_engine.phi[6]))
            # Map φ₆ ∈ [-∞, ∞] to [0.5, 1.5] via sigmoid
            field_factor = 0.5 + 1.0 / (1.0 + np.exp(-phi_6_mean))
        
        # # Technical implementation
        # # Implementation
        if success:
            learning_rate = min(
                self.base_learning_rate * (1.1 ** current_streak) * field_factor,
                self.base_learning_rate * 10.0  # 10x max speedup
            )
        else:
            # # Technical implementation
            # # Implementation
            learning_rate = self.base_learning_rate * field_factor
        
        # # Technical implementation
        # # Implementation
        self.memory.record_inference(success, learning_rate)
        
        # # Technical implementation
        # # Implementation
        return np.array([learning_rate, field_factor, float(success)])


class OscillatorModule:
    """
    Stage 5: Apply temporal oscillatory modulation with IFT field substrate.
    
    # Technical implementation
    The phase rotates like a cosmic clock, breathing life into static patterns.
    Beneath the rhythm, ten fields dance in twelve-fold harmony—a substrate
    of cosmic forces that modulates the very fabric of cognition. The oscillator
    pulses with adaptive thresholds and leak rates, self-regulating through
    Atlantean wisdom embedded in the field topology.
    
    # Implementation
    Applies multiplicative modulation: output = signal * (1 + α*sin(φ))
    where φ advances linearly at Δφ = 0.1 rad/inference, creating periodic
    amplitude variation with period T ≈ 63 steps. Phase persists across
    requests (stateful continuity).
    
    Enhanced with IFT substrate:
    - 10 coupled fields evolve via PDEs with 12-fold symmetry
    - Adaptive threshold: θ(x) = base - 0.5·φ₀(x)
    - Adaptive leak: λ(x) = base/(1 + exp(φ₄(x)))
    - Field metrics tracked: energy, symmetry order, coupling
    
    Mathematical properties:
    - Phase domain: φ ∈ [0, 2π) (periodic, wraps)
    - Modulation amplitude: α = 0.3 (30% variation)
    - Output range: signal * [0.7, 1.3]
    - Period: 2π/0.1 ≈ 62.83 inferences
    - Field evolution: ∂φₙ/∂t = Fₙ(φₙ, ∇²φₙ, ∇⁴φₙ) + ε·coupling
    """
    
    def __init__(self, memory: MemoryStore, enable_ift: bool = True, 
                 field_shape: tuple = (32, 32)):
        self.memory = memory
        self.enable_ift = enable_ift and memory.enable_ift
        
        # Initialize IFT field substrate if enabled
        if self.enable_ift:
            # If we have saved state, use its shape; otherwise use default
            actual_shape = field_shape
            if memory.ift_field_state is not None:
                # Extract shape from saved state: (10, H, W)
                saved_shape = memory.ift_field_state.shape[1:]
                if len(saved_shape) == 2:
                    actual_shape = saved_shape
                    logger.info(f"Using field shape from saved state: {actual_shape}")
            
            self.field_engine = FieldEngine(
                shape=actual_shape,
                mode="2d",
                eps=0.15
            )
            # Restore previous state if available
            if memory.ift_field_state is not None:
                try:
                    self.field_engine.phi = memory.ift_field_state
                    logger.info(f"Restored IFT field state: {memory.ift_field_state.shape}")
                except Exception as e:
                    logger.warning(f"Failed to restore IFT state: {e}")
        else:
            self.field_engine = None
    
    def modulate(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Modulate signal by current oscillatory phase and IFT field substrate.
        
        # Technical implementation
        # Implementation
        """
        # # Technical implementation
        # # Implementation
        phase = self.memory.oscillator_phase
        
        # # Technical implementation
        # # Implementation
        modulation = torch.sin(torch.tensor(phase)).item()
        
        # # Technical implementation
        # # Implementation
        self.memory.oscillator_phase = (phase + 0.1) % (2 * np.pi)
        
        # # Technical implementation
        # # Implementation
        if self.enable_ift and self.field_engine is not None:
            # Evolve fields
            self.field_engine.step(dt=0.01)
            
            # Extract metrics (convert arrays to scalars)
            potential = float(np.mean(self.field_engine.Phi))  # Global average
            symmetry = float(self.field_engine.get_symmetry_order())
            energy = float(self.field_engine.get_field_energy())
            
            # Update memory state
            self.memory.update_ift_state(
                field_state=np.array([f.copy() for f in self.field_engine.phi]),
                potential=potential,
                symmetry=symmetry,
                energy=energy
            )
            
            # # Technical implementation
            # # Implementation
            # Map Φ ∈ [-∞, ∞] to [0.2, 0.4] via tanh
            field_modulation = 0.3 + 0.1 * np.tanh(potential)
        else:
            field_modulation = 0.3  # Default amplitude
        
        # # Technical implementation
        # # Implementation
        modulated = input_signal * (1.0 + field_modulation * modulation)
        
        return modulated
    
    def get_adaptive_threshold(self, base_threshold: float = 1.0) -> float:
        """
        Get field-modulated firing threshold.
        
        # Technical implementation
        # Implementation
        """
        if self.enable_ift and self.field_engine is not None:
            phi_0_mean = float(np.mean(self.field_engine.phi[0]))
            return base_threshold - 0.5 * phi_0_mean
        return base_threshold
    
    def get_adaptive_leak(self, base_leak: float = 0.1) -> float:
        """
        Get field-modulated leak rate.
        
        # Technical implementation
        # Implementation
        """
        if self.enable_ift and self.field_engine is not None:
            phi_4_mean = float(np.mean(self.field_engine.phi[4]))
            return base_leak / (1.0 + np.exp(phi_4_mean))
        return base_leak


class SymbolicReasoner:
    """Stage 6: Symbolic interpretation of neural output."""
    
    def __init__(self, memory: MemoryStore, use_hrm: bool = True):
        self.memory = memory
        self.use_hrm = use_hrm
        self._relation_count = 0
        self.hrm_engine = SimulationEngine(field_shape=(12, 12)) if use_hrm else None
        self.hrm_state: Optional[HRMState] = self.hrm_engine.initialize() if self.hrm_engine else None
        self._hrm_stride = 3
        self._last_hrm_metrics: Dict[str, float] = {}
    
    def _inject_signal(self, neural_output: torch.Tensor, concepts: List[str], text: str):
        if not self.hrm_state:
            return
        signal = self.hrm_state.state_vector.copy()
        signal[0] = float(torch.norm(neural_output).item())
        signal[1] = float(neural_output.mean().item()) if neural_output.numel() > 0 else 0.0
        signal[2] = float(len(concepts)) / 10.0
        signal[3] = float(len(text)) / 256.0
        signal[4] = float(sum(ord(ch) for ch in text[:32]) % 997) / 997.0 if text else 0.0
        self.hrm_state.state_vector = 0.85 * self.hrm_state.state_vector + 0.15 * signal
    
    def get_hrm_snapshot(self) -> Dict[str, Any]:
        if not self.hrm_state:
            return {'enabled': False}
        return {
            'enabled': True,
            'theta': float(self.hrm_state.theta),
            'channel': int(self.hrm_state.channel),
            'domain': int(self.hrm_state.domain),
            'layer': int(self.hrm_state.layer),
            'guna': self.hrm_state.guna,
            'step_count': int(self.hrm_state.step_count),
            'objective': float(self.hrm_state.last_metrics.get('objective', 0.0)),
            'coherence': float(self.hrm_state.last_metrics.get('f_coherence', 0.0)),
            'stability': float(self.hrm_state.last_metrics.get('f_stability', 0.0)),
        }
    
    def interpret(self, neural_output: torch.Tensor, concepts: List[str], text: str = "") -> str:
        """
        Derive symbolic meaning from neural activations.
        
        Combines algebraic reasoning, logic, and knowledge graphs.
        """
        try:
            # Keep concept memory updates cheap and deterministic.
            normalized_concepts = [c.strip().lower() for c in concepts if c and c.strip()]
            for concept in normalized_concepts:
                self.memory.add_concept(concept)
            if len(normalized_concepts) > 1:
                self._relation_count += len(normalized_concepts) - 1

            # HRM-backed symbolic synthesis.
            hrm_metrics: Dict[str, float] = {}
            hrm_stepped = False
            if self.use_hrm and self.hrm_engine and self.hrm_state is not None:
                self._inject_signal(neural_output, normalized_concepts, text)
                if self.hrm_state.step_count % self._hrm_stride == 0:
                    self.hrm_state, hrm_metrics = self.hrm_engine.step(self.hrm_state, dt=0.02)
                    self._last_hrm_metrics = hrm_metrics
                    hrm_stepped = True
                else:
                    self.hrm_engine.cycle_engine.update(self.hrm_state, dt=1.0)
                    self.hrm_engine.topology.route(self.hrm_state)
                    self.hrm_state.step_count += 1
                    hrm_metrics = self._last_hrm_metrics

            neural_magnitude = torch.norm(neural_output).item()
            if self.hrm_state is not None:
                manifold_summary = (
                    f"HRM(theta={self.hrm_state.theta:.3f}, ch={self.hrm_state.channel}, "
                    f"d={self.hrm_state.domain}, l={self.hrm_state.layer})"
                )
            else:
                manifold_summary = "HRM(disabled)"

            coherence = hrm_metrics.get('f_coherence', 0.0)
            stability = hrm_metrics.get('f_stability', 0.0)
            objective = hrm_metrics.get('objective', 0.0)

            result = (
                f"{manifold_summary} | concepts={len(normalized_concepts)} "
                f"relations={self._relation_count} | "
                f"coherence={coherence:.3f} stability={stability:.3f} "
                f"objective={objective:.3f} | activation={neural_magnitude:.4f}"
            )
            
            # Keep trace payload compact and write less frequently in HRM mode.
            if not self.use_hrm or hrm_stepped:
                self.memory.add_reasoning_trace({
                    'concepts': normalized_concepts,
                    'neural_magnitude': neural_magnitude,
                    'relation_count': self._relation_count,
                    'hrm_channel': int(self.hrm_state.channel) if self.hrm_state else 0,
                    'hrm_objective': float(objective),
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Symbolic reasoning failed: {e}")
            return f"Reasoning error: {str(e)}"


class StatefulSymbolicPredictiveInterpreter:
    """
    Complete 8-stage stateful inference system with IFT field substrate.
    
    Implements the full pipeline with persistent memory, governance, and
    ten-field substrate for adaptive parameter modulation.
    """
    
    def __init__(self, model_version: str = "spi-symbolic-0.2.0", 
                 enable_ift: bool = True, field_shape: tuple = (32, 32),
                 enable_hrm: bool = True):
        self.model_version = model_version
        self.device = torch.device('cpu')
        self.enable_ift = enable_ift
        
        # Initialize memory (stateful across requests)
        self.memory = MemoryStore(enable_ift=enable_ift)
        
        # Initialize pipeline stages
        self.encoder = InputEncoder(input_dim=128)
        self.pattern_extractor = PatternExtractor(n_patterns=3)
        self.spike_generator = SpikeGenerator(input_dim=128, hidden_dim=64)
        self.oscillator = OscillatorModule(self.memory, enable_ift=enable_ift, field_shape=field_shape)
        self.neuroplastic_adapter = NeuroplasticAdapter(self.memory, oscillator=self.oscillator)
        self.symbolic_reasoner = SymbolicReasoner(self.memory, use_hrm=enable_hrm)
        
        # Governance
        self.policy_engine = PolicyEngine()
        self.governance_service = GovernanceService(self.policy_engine)

        self.capability_profiles = {
            'general': {'coding_bias': 0.0, 'tool_bias': 0.0, 'abstraction_bias': 1.0},
            'coding': {'coding_bias': 0.25, 'tool_bias': 0.10, 'abstraction_bias': 0.85},
            'tool_use': {'coding_bias': 0.10, 'tool_bias': 0.30, 'abstraction_bias': 0.80},
        }
        self._cross_reference_rounds = 3
        
        logger.info(f"StatefulSymbolicPredictiveInterpreter initialized (v{self.model_version}, IFT={enable_ift})")
    
    async def process(self, input_data: Dict[str, Any], request_id: str = "") -> Dict[str, Any]:
        """
        Full 8-stage pipeline execution with state management and governance.
        
        Args:
            input_data: Dict with 'text' or 'vector' or 'concepts'
            request_id: For tracing
        
        Returns:
            Complete output with governance applied
        """
        ctx = StatefulInferenceContext(self.memory, request_id)
        normalized_input = self._normalize_input_data(input_data)
        
        try:
            # Stage 1: Encode Input
            ctx.start_stage("encode")
            if 'text' in normalized_input:
                ctx.encoded_input = self.encoder.encode_text(normalized_input['text'])
            elif 'vector' in normalized_input:
                ctx.encoded_input = self.encoder.encode_vector(normalized_input['vector'])
            else:
                ctx.encoded_input = np.zeros(128)
            ctx.end_stage("encode")
            
            # Stage 2: Extract Patterns
            ctx.start_stage("pattern_extraction")
            ctx.patterns, pattern_confidence = self.pattern_extractor.extract(ctx.encoded_input)
            ctx.end_stage("pattern_extraction")
            
            # Stage 3: Generate Spikes
            ctx.start_stage("spike_generation")
            input_tensor = torch.from_numpy(ctx.encoded_input).float().to(self.device)
            ctx.spikes = self.spike_generator.forward(input_tensor, num_steps=5)
            ctx.end_stage("spike_generation")
            
            # Stage 4: Neuroplastic Adaptation
            ctx.start_stage("neuroplasticity")
            success = pattern_confidence > 0.5  # Simple success criterion
            neuroplastic_signal = self.neuroplastic_adapter.adapt(success)
            ctx.neuroplastic_update = neuroplastic_signal
            ctx.end_stage("neuroplasticity")
            
            # Stage 5: Oscillatory Modulation
            ctx.start_stage("oscillation")
            ctx.oscillated_output = self.oscillator.modulate(ctx.spikes)
            ctx.end_stage("oscillation")
            
            # Stage 6: Symbolic Interpretation
            ctx.start_stage("symbolic_reasoning")
            concepts = normalized_input.get('concepts', [])
            ctx.symbolic_interpretation = await self._async_symbolic_reasoning(
                ctx.oscillated_output, concepts, normalized_input.get('text', '')
            )
            ctx.end_stage("symbolic_reasoning")

            # Stage 6b: Cross-reference generation across field functions
            ctx.start_stage("cross_reference_generation")
            generation_payload = self._cross_reference_generation(
                base_symbolic=ctx.symbolic_interpretation,
                pattern_confidence=pattern_confidence,
                concepts=concepts,
                text=normalized_input.get('text', ''),
                mode=normalized_input.get('mode', 'general'),
            )
            ctx.end_stage("cross_reference_generation")
            
            # Stage 7: Governance Evaluation
            ctx.start_stage("governance")
            policy_context = PolicyContext(
                input_text=normalized_input.get('text', ''),
                neural_magnitude=float(torch.norm(ctx.spikes).item()),
                pattern_confidence=pattern_confidence,
                symbolic_concepts=concepts,
                request_id=request_id,
            )
            ctx.governance_decision = self.policy_engine.evaluate(policy_context).audit_entry
            ctx.end_stage("governance")
            
            # Stage 8: Output Synthesis
            ctx.start_stage("output_synthesis")
            ctx.final_output = self._synthesize_output(
                ctx,
                generation_payload=generation_payload,
                mode=normalized_input.get('mode', 'general'),
            )
            ctx.end_stage("output_synthesis")
            
            # Apply governance conditioning
            ctx.final_output = self.governance_service.evaluate_and_condition_output(
                ctx.final_output,
                policy_context
            )
            
            # Update memory context
            self.memory.add_to_context({
                'concept': ' '.join(concepts),
                'output': ctx.final_output.get('symbolic_result', '')[:80],
            })
            
            return ctx.final_output
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return self._error_output(str(e), ctx)
    
    async def _async_symbolic_reasoning(self, neural_output: torch.Tensor, 
                                       concepts: List[str], text: str = "") -> str:
        """Async wrapper for symbolic reasoning."""
        return await asyncio.to_thread(
            self.symbolic_reasoner.interpret,
            neural_output,
            concepts,
            text,
        )
    
    def _synthesize_output(
        self,
        ctx: StatefulInferenceContext,
        generation_payload: Optional[Dict[str, Any]] = None,
        mode: str = 'general',
    ) -> Dict[str, Any]:
        """Combine all pipeline outputs into final response."""
        metrics = self.memory.get_neuroplastic_metrics()

        symbolic_result = ctx.symbolic_interpretation or "No interpretation"
        if generation_payload and generation_payload.get('best_response'):
            symbolic_result = generation_payload['best_response']

        world_knowledge_seed = self._build_world_knowledge_seed(limit=8)
        
        return {
            'request_id': ctx.request_id,
            'model_version': self.model_version,
            'symbolic_result': symbolic_result,
            'neural_magnitude': float(torch.norm(ctx.spikes).item()) if ctx.spikes is not None else 0.0,
            'spike_rate': float(ctx.spikes.mean().item()) if ctx.spikes is not None else 0.0,
            'oscillatory_phase': float(self.memory.oscillator_phase),
            'syntropy_state': self.memory.syntropy_values.copy(),
            'neuroplastic_metrics': metrics,
            'ift_field_metrics': self.memory.get_ift_metrics() if self.enable_ift else {'enabled': False},
            'adaptive_parameters': {
                'threshold': self.oscillator.get_adaptive_threshold() if self.enable_ift else 1.0,
                'leak': self.oscillator.get_adaptive_leak() if self.enable_ift else 0.1,
            },
            'stage_times': ctx.stage_times,
            'memory_state': {
                'total_concepts': len(self.memory.concept_history),
                'reasoning_traces': len(self.memory.reasoning_traces),
                'success_streak': metrics['success_streak'],
            },
            'hrm_state': self.symbolic_reasoner.get_hrm_snapshot(),
            'generation_mode': mode,
            'generation_result': symbolic_result,
            'field_cross_reference': generation_payload or {
                'winning_field': 'symbolic_baseline',
                'scores': {'symbolic_baseline': 0.0},
                'consensus_score': 0.0,
            },
            'neural_core': {
                'world_knowledge_seed': world_knowledge_seed,
                'context_threads': self.memory.get_context_summary().get('recent_concepts', []),
                'coherence_anchor': float(metrics.get('success_streak', 0)) / 10.0,
            },
        }

    def _normalize_input_data(self, input_data: Any) -> Dict[str, Any]:
        """Normalize messy inputs into stable schema for robust processing."""
        if isinstance(input_data, dict):
            payload: Dict[str, Any] = dict(input_data)
        elif isinstance(input_data, str):
            payload = {'text': input_data}
        elif isinstance(input_data, (list, tuple)):
            payload = {'concepts': [str(x) for x in input_data]}
        else:
            payload = {'text': str(input_data)}

        raw_text = payload.get('text', '')
        text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", str(raw_text))
        text = re.sub(r"\s+", " ", text).strip()

        concepts_raw = payload.get('concepts', [])
        if isinstance(concepts_raw, str):
            concepts = [c.strip().lower() for c in re.split(r"[,;|]", concepts_raw) if c.strip()]
        elif isinstance(concepts_raw, (list, tuple, set)):
            concepts = [str(c).strip().lower() for c in concepts_raw if str(c).strip()]
        else:
            concepts = []

        if not concepts and text:
            tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_-]+", text) if len(t) > 3]
            concepts = tokens[:8]

        mode = str(payload.get('mode', 'general')).strip().lower()
        if mode not in self.capability_profiles:
            mode = 'general'

        normalized: Dict[str, Any] = {
            'text': text,
            'concepts': list(dict.fromkeys(concepts)),
            'mode': mode,
        }

        if 'vector' in payload:
            normalized['vector'] = payload['vector']

        return normalized

    def _cross_reference_generation(
        self,
        base_symbolic: str,
        pattern_confidence: float,
        concepts: List[str],
        text: str,
        mode: str,
    ) -> Dict[str, Any]:
        """Generate responses via field-based candidates and pick consensus best."""
        profile = self.capability_profiles.get(mode, self.capability_profiles['general'])
        hrm = self.symbolic_reasoner.get_hrm_snapshot()
        coherence = float(hrm.get('coherence', 0.0)) if hrm.get('enabled') else 0.0
        stability = float(hrm.get('stability', 0.0)) if hrm.get('enabled') else 0.0
        success_rate = float(self.memory.get_neuroplastic_metrics().get('success_rate', 0.0))
        context = self.memory.get_context_summary()
        recent_concepts = context.get('recent_concepts', []) if isinstance(context, dict) else []
        overlap = self._concept_overlap(concepts, recent_concepts)

        coding_hint = " Prioritize executable structure and testability." if mode == 'coding' else ""
        tool_hint = " Prefer explicit tool/action sequencing." if mode == 'tool_use' else ""

        candidates: List[Dict[str, Any]] = [
            {
                'field': 'symbolic_field',
                'response': base_symbolic,
                'score': 0.35 + 0.35 * pattern_confidence + 0.10 * overlap,
            },
            {
                'field': 'coherence_field',
                'response': (
                    f"Coherent synthesis across concepts {concepts[:5]} with continuity from recent context "
                    f"{recent_concepts[-3:]}."
                ),
                'score': 0.25 + 0.40 * coherence + 0.20 * stability + 0.10 * overlap,
            },
            {
                'field': 'abstraction_transfer_field',
                'response': (
                    "Transfer abstractions by recursively mapping concept relations into reusable reasoning "
                    f"patterns across domains.{coding_hint}{tool_hint}"
                ),
                'score': 0.20 + 0.30 * profile['abstraction_bias'] + 0.20 * pattern_confidence + 0.10 * overlap,
            },
            {
                'field': 'capability_mode_field',
                'response': (
                    f"Mode={mode}: blend field evidence, memory continuity, and governance-safe synthesis "
                    f"for robust output under noisy inputs.{coding_hint}{tool_hint}"
                ),
                'score': 0.15 + 0.25 * profile['coding_bias'] + 0.25 * profile['tool_bias'] + 0.20 * success_rate,
            },
        ]

        for _ in range(max(1, self._cross_reference_rounds - 1)):
            best = max(candidates, key=lambda c: c['score'])
            for candidate in candidates:
                alignment = self._token_overlap(best['response'], candidate['response'])
                candidate['score'] = 0.75 * candidate['score'] + 0.25 * alignment

        best = max(candidates, key=lambda c: c['score'])
        scores = {c['field']: float(c['score']) for c in candidates}

        return {
            'winning_field': best['field'],
            'scores': scores,
            'consensus_score': float(best['score']),
            'best_response': best['response'],
        }

    def _concept_overlap(self, a: List[str], b: List[str]) -> float:
        """Overlap ratio between concept collections."""
        set_a = {str(x).strip().lower() for x in a if str(x).strip()}
        set_b = {str(x).strip().lower() for x in b if str(x).strip()}
        if not set_a or not set_b:
            return 0.0
        return len(set_a.intersection(set_b)) / max(len(set_a), len(set_b))

    def _token_overlap(self, a: str, b: str) -> float:
        """Lexical overlap score for recursive cross-reference rounds."""
        tokens_a = {t for t in re.findall(r"[a-zA-Z0-9_-]+", str(a).lower()) if len(t) > 2}
        tokens_b = {t for t in re.findall(r"[a-zA-Z0-9_-]+", str(b).lower()) if len(t) > 2}
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a.intersection(tokens_b)) / max(len(tokens_a), len(tokens_b))

    def _build_world_knowledge_seed(self, limit: int = 8) -> List[str]:
        """Build compact world-knowledge seed from accumulated concept history."""
        counts: Dict[str, int] = {}
        for concept in self.memory.concept_history:
            key = str(concept).strip().lower()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1

        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [name for name, _ in ordered[:limit]]
    
    def _error_output(self, error_msg: str, ctx: StatefulInferenceContext) -> Dict[str, Any]:
        """Generate error response."""
        return {
            'request_id': ctx.request_id,
            'model_version': self.model_version,
            'error': error_msg,
            'stage_times': ctx.stage_times,
        }
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Export current memory state for inspection."""
        return {
            'oscillator_phase': self.memory.oscillator_phase,
            'syntropy_values': self.memory.syntropy_values,
            'neuroplastic_metrics': self.memory.get_neuroplastic_metrics(),
            'concept_count': len(self.memory.concept_history),
            'recent_concepts': self.memory.concept_history[-10:],
            'context_summary': self.memory.get_context_summary(),
        }
