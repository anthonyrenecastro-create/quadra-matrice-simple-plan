#!/usr/bin/env python3
"""
CognitionSim - Cognition Demonstration
Interactive demo to observe and understand the cognitive system in real-time.

This script demonstrates:
1. Spiking Neural Field Dynamics - Watch neurons fire and spike
2. Neuroplastic Memory - See how the system learns and remembers
3. Symbolic Reasoning - Observe logical inference and knowledge building
4. Oscillatory Dynamics - Track field coherence and resonance
"""

import asyncio
import json
import time
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List
from collections import defaultdict
import sys

# Configure logging for visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from atlantean_hrm_spi import (
        OscillatorySynapseTheory,
        NeuroplasticityManager,
        SymbolicPredictiveInterpreter,
        SymbolicConfig
    )
except ImportError:
    logger.error("Could not import CognitionSim components. Ensure atlantean_hrm_spi.py is available.")
    sys.exit(1)


# ============================================================================
# COGNITION OBSERVER - Real-time cognitive process tracking
# ============================================================================

class CognitionObserver:
    """Observes and reports on cognitive processes"""
    
    def __init__(self):
        self.observations = defaultdict(list)
        self.spike_history = []
        self.field_coherence_history = []
        self.memory_state_history = []
        self.reasoning_traces = []
        
    def record_spike_activity(self, spikes: torch.Tensor, neuron_ids: List[int] = None):
        """Record when neurons fire"""
        if spikes.numel() == 0:
            return
        
        num_spikes = int(spikes.sum().item())
        firing_rate = float(spikes.mean().item()) * 100
        
        record = {
            'timestamp': time.time(),
            'spike_count': num_spikes,
            'firing_rate_percent': firing_rate,
            'active_neurons': num_spikes
        }
        self.spike_history.append(record)
        self.observations['spikes'].append(record)
        
        return record
    
    def record_field_coherence(self, field_state: torch.Tensor):
        """Track field coherence and stability"""
        if field_state.numel() == 0:
            return
        
        field_mean = float(field_state.mean().item())
        field_std = float(field_state.std().item())
        field_energy = float((field_state ** 2).sum().item())
        
        record = {
            'timestamp': time.time(),
            'mean': field_mean,
            'std': field_std,
            'energy': field_energy,
            'coherence': 1.0 - min(1.0, field_std)  # Higher std = lower coherence
        }
        self.field_coherence_history.append(record)
        self.observations['coherence'].append(record)
        
        return record
    
    def record_memory_consolidation(self, memory_state: torch.Tensor, consolidation_rate: float):
        """Track memory storage and consolidation"""
        if memory_state is None:
            return
        
        memory_magnitude = float((memory_state ** 2).sum().sqrt().item())
        
        record = {
            'timestamp': time.time(),
            'magnitude': memory_magnitude,
            'consolidation_rate': consolidation_rate,
            'dimensionality': memory_state.shape[-1] if hasattr(memory_state, 'shape') else 0
        }
        self.memory_state_history.append(record)
        self.observations['memory'].append(record)
        
        return record
    
    def record_symbolic_reasoning(self, query: str, concepts: List[str], result: str):
        """Track symbolic reasoning processes"""
        record = {
            'timestamp': time.time(),
            'query': query,
            'concepts': concepts,
            'result': result,
            'num_concepts': len(concepts)
        }
        self.reasoning_traces.append(record)
        self.observations['reasoning'].append(record)
        
        return record
    
    def print_summary(self):
        """Print a summary of observed cognition"""
        print("\n" + "="*70)
        print("COGNITIVE PROCESS SUMMARY")
        print("="*70)
        
        if self.spike_history:
            avg_spikes = np.mean([x['spike_count'] for x in self.spike_history])
            max_rate = max(x['firing_rate_percent'] for x in self.spike_history)
            print(f"\n🧠 NEURAL FIRING:")
            print(f"   - Average active neurons: {avg_spikes:.1f}")
            print(f"   - Peak firing rate: {max_rate:.2f}%")
            print(f"   - Total firing events: {len(self.spike_history)}")
        
        if self.field_coherence_history:
            avg_coherence = np.mean([x['coherence'] for x in self.field_coherence_history])
            print(f"\n📡 FIELD DYNAMICS:")
            print(f"   - Average coherence: {avg_coherence:.3f}")
            print(f"   - Field observations: {len(self.field_coherence_history)}")
        
        if self.memory_state_history:
            avg_memory = np.mean([x['magnitude'] for x in self.memory_state_history])
            print(f"\n💾 MEMORY CONSOLIDATION:")
            print(f"   - Average memory magnitude: {avg_memory:.4f}")
            print(f"   - Consolidation events: {len(self.memory_state_history)}")
        
        if self.reasoning_traces:
            print(f"\n⚡ SYMBOLIC REASONING:")
            print(f"   - Reasoning processes: {len(self.reasoning_traces)}")
            for i, trace in enumerate(self.reasoning_traces[:3], 1):
                print(f"   - Process {i}: Concepts={trace['num_concepts']}")
        
        print("\n" + "="*70)


# ============================================================================
# INTERACTIVE COGNITION DEMO
# ============================================================================

class CognitionDemo:
    """Interactive demonstration of CognitionSim cognition"""
    
    def __init__(self, field_size: int = 100, device: str = 'cpu'):
        self.field_size = field_size
        self.device = device
        self.observer = CognitionObserver()
        
        logger.info(f"Initializing CognitionSim Cognition Demo (field_size={field_size}, device={device})")
        
        # Initialize core cognitive components
        self.oscillator = OscillatorySynapseTheory(field_size=field_size, device=device)
        logger.info("✓ Oscillatory Synapse Theory Engine initialized")
        
        # Create mock objects for compatibility
        class MockCoreField:
            def __init__(self, field):
                self.field = field
            def get_state(self):
                return self.field
            def update_with_vibrational_mode(self, data):
                pass
        
        class MockPatternModule:
            pass
        
        # Symbolic reasoning
        config = SymbolicConfig()
        self.mock_core_field = MockCoreField(self.oscillator.field)
        self.mock_pattern = MockPatternModule()
        
        self.symbolic_interpreter = SymbolicPredictiveInterpreter(
            pattern_module=self.mock_pattern,
            core_field=self.mock_core_field,
            config=config
        )
        logger.info("✓ Symbolic Predictive Interpreter initialized")
    
    def observe_neural_spiking(self, num_iterations: int = 5):
        """Demonstrate neural spiking dynamics"""
        print("\n" + "="*70)
        print("🧠 NEURAL SPIKING DEMONSTRATION")
        print("="*70)
        print("Observing spiking neural field dynamics...\n")
        
        for i in range(num_iterations):
            # Generate synthetic input
            input_data = torch.randn(self.field_size, device=self.device)
            
            # Process through spiking networks
            spikes_update = self.oscillator.nn1(input_data)
            spikes_syntropy = self.oscillator.nn2(input_data)
            spikes_feedback = self.oscillator.nn3(input_data)
            
            # Record observations
            update_obs = self.observer.record_spike_activity(spikes_update)
            syntropy_obs = self.observer.record_spike_activity(spikes_syntropy)
            feedback_obs = self.observer.record_spike_activity(spikes_feedback)
            
            print(f"Iteration {i+1}:")
            print(f"  🔴 Field Update Network: {update_obs['active_neurons']} neurons firing ({update_obs['firing_rate_percent']:.2f}%)")
            print(f"  🟡 Syntropy Network:     {syntropy_obs['active_neurons']} neurons firing ({syntropy_obs['firing_rate_percent']:.2f}%)")
            print(f"  🟢 Feedback Network:     {feedback_obs['active_neurons']} neurons firing ({feedback_obs['firing_rate_percent']:.2f}%)")
            print()
            
            time.sleep(0.2)
        
        print("✓ Neural spiking demonstration complete\n")
    
    def observe_field_coherence(self, num_steps: int = 5):
        """Demonstrate field coherence evolution"""
        print("="*70)
        print("📡 FIELD COHERENCE DEMONSTRATION")
        print("="*70)
        print("Tracking field state coherence and stability...\n")
        
        field_state = self.oscillator.field.clone()
        
        for step in range(num_steps):
            # Evolve field
            input_perturbation = torch.randn_like(field_state) * 0.1
            field_state = field_state + input_perturbation
            field_state = torch.clamp(field_state, -1, 1)
            
            # Record coherence
            coh_obs = self.observer.record_field_coherence(field_state)
            
            print(f"Step {step+1}:")
            print(f"  Field Mean:      {coh_obs['mean']:7.4f}")
            print(f"  Field Std Dev:   {coh_obs['std']:7.4f}")
            print(f"  Coherence Score: {coh_obs['coherence']:.3f} {'🔒 HIGH' if coh_obs['coherence'] > 0.7 else '❓ MEDIUM' if coh_obs['coherence'] > 0.4 else '🌪️  LOW'}")
            print()
            
            time.sleep(0.2)
        
        print("✓ Field coherence demonstration complete\n")
    
    def observe_memory_consolidation(self):
        """Demonstrate neuroplastic memory consolidation"""
        print("="*70)
        print("💾 MEMORY CONSOLIDATION DEMONSTRATION")
        print("="*70)
        print("Observing exponential memory consolidation through time...\n")
        
        memory = None
        learning_rate = 0.1
        consolidation_factor = 0.9
        
        for step in range(5):
            # Simulate new experience
            new_experience = torch.randn(self.field_size, device=self.device)
            
            # Consolidate into memory
            if memory is None:
                memory = new_experience.clone()
                consolidation_update = learning_rate
            else:
                memory = consolidation_factor * memory + (1 - consolidation_factor) * new_experience
                consolidation_update = consolidation_factor
            
            mem_obs = self.observer.record_memory_consolidation(memory, consolidation_update)
            
            print(f"Experience {step+1}:")
            print(f"  Memory Magnitude:        {mem_obs['magnitude']:.4f}")
            print(f"  Consolidation Rate:      {consolidation_update:.3f}")
            print(f"  Memory Dimensionality:   {mem_obs['dimensionality']}")
            print(f"  Integration Progress:    {'█' * int(consolidation_update * 10)}{'░' * (10 - int(consolidation_update * 10))} {consolidation_update*100:.1f}%")
            print()
            
            time.sleep(0.3)
        
        print("✓ Memory consolidation demonstration complete\n")
    
    def observe_symbolic_reasoning(self):
        """Demonstrate symbolic reasoning and knowledge building"""
        print("="*70)
        print("⚡ SYMBOLIC REASONING DEMONSTRATION")
        print("="*70)
        print("Observing symbolic inference and knowledge graph construction...\n")
        
        test_concepts = [
            ["intelligence", "learning", "adaptation"],
            ["neural", "network", "computation"],
            ["memory", "consolidation", "recall"],
            ["knowledge", "reasoning", "inference"]
        ]
        
        for i, concepts in enumerate(test_concepts, 1):
            print(f"Reasoning Process {i}:")
            print(f"  Input Concepts: {concepts}")
            
            # Simulate reasoning
            neural_output = torch.randn(self.field_size, device=self.device)
            
            # Build knowledge graph
            self.symbolic_interpreter.build_knowledge_graph(concepts, neural_output)
            graph_query = self.symbolic_interpreter.query_knowledge_graph()
            
            reasoning_result = f"Inferred relationships: {graph_query}"
            
            # Record reasoning
            self.observer.record_symbolic_reasoning(
                f"Analyze concepts: {concepts}",
                concepts,
                reasoning_result
            )
            
            print(f"  {reasoning_result}")
            print()
            
            time.sleep(0.2)
        
        print("✓ Symbolic reasoning demonstration complete\n")
    
    def observe_integrated_cognition(self):
        """Demonstrate integrated cognitive processing"""
        print("="*70)
        print("🎯 INTEGRATED COGNITION DEMONSTRATION")
        print("="*70)
        print("Observing all cognitive systems working together...\n")
        
        test_inputs = [
            "learning and adaptation",
            "neural computation",
            "memory formation"
        ]
        
        for i, text_input in enumerate(test_inputs, 1):
            print(f"Cognitive Process {i}: Processing '{text_input}'")
            print()
            
            # Step 1: Process streaming data
            print("  Step 1️⃣  - Neural Processing")
            feature_vector = self.oscillator.process_streamed_data(text_input)
            spikes = (torch.randn(self.field_size, device=self.device) > 0.5).float()
            spike_obs = self.observer.record_spike_activity(spikes)
            print(f"         Neurons activated: {spike_obs['active_neurons']}")
            
            # Step 2: Update field state
            print("  Step 2️⃣  - Field Evolution")
            field_state = self.oscillator.field.clone()
            coh_obs = self.observer.record_field_coherence(field_state)
            print(f"         Field coherence: {coh_obs['coherence']:.3f}")
            
            # Step 3: Memory consolidation
            print("  Step 3️⃣  - Memory Update")
            mem_obs = self.observer.record_memory_consolidation(feature_vector, 0.9)
            print(f"         Memory magnitude: {mem_obs['magnitude']:.4f}")
            
            # Step 4: Symbolic reasoning
            print("  Step 4️⃣  - Symbolic Reasoning")
            concepts = text_input.split()
            self.observer.record_symbolic_reasoning(f"Process: {text_input}", concepts, "Reasoning complete")
            print(f"         Concepts analyzed: {len(concepts)}")
            
            print()
            time.sleep(0.5)
        
        print("✓ Integrated cognition demonstration complete\n")
    
    def run_interactive_menu(self):
        """Interactive menu for exploring cognition"""
        while True:
            print("\n" + "="*70)
            print("QUADRA MATRIX COGNITION EXPLORER")
            print("="*70)
            print("\nChoose what to observe:")
            print("  1. 🧠 Neural Spiking - Watch neurons fire and spike")
            print("  2. 📡 Field Coherence - Track field state and stability")
            print("  3. 💾 Memory Consolidation - See exponential memory formation")
            print("  4. ⚡ Symbolic Reasoning - Observe logical inference")
            print("  5. 🎯 Integrated Cognition - All systems working together")
            print("  6. 📊 Show Summary - Display all observations")
            print("  7. 🚀 Full Sequence - Run everything in order")
            print("  8. 🚪 Exit")
            print()
            
            choice = input("Enter your choice (1-8): ").strip()
            
            try:
                if choice == '1':
                    self.observe_neural_spiking(num_iterations=5)
                elif choice == '2':
                    self.observe_field_coherence(num_steps=5)
                elif choice == '3':
                    self.observe_memory_consolidation()
                elif choice == '4':
                    self.observe_symbolic_reasoning()
                elif choice == '5':
                    self.observe_integrated_cognition()
                elif choice == '6':
                    self.observer.print_summary()
                elif choice == '7':
                    print("\nRunning complete cognitive demonstration sequence...\n")
                    self.observe_neural_spiking(num_iterations=3)
                    time.sleep(0.5)
                    self.observe_field_coherence(num_steps=3)
                    time.sleep(0.5)
                    self.observe_memory_consolidation()
                    time.sleep(0.5)
                    self.observe_symbolic_reasoning()
                    time.sleep(0.5)
                    self.observe_integrated_cognition()
                    self.observer.print_summary()
                elif choice == '8':
                    print("\n👋 Thank you for exploring CognitionSim cognition!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-8.")
            except KeyboardInterrupt:
                print("\n\n⏸️  Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during demonstration: {e}", exc_info=True)
                print(f"❌ Error: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "🌟 QUADRA MATRIX COGNITION DEMO 🌟" + " "*19 + "║")
    print("║" + " "*10 + "See the cognitive system in action!" + " "*23 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    print(f"PyTorch Version: {torch.__version__}\n")
    
    # Create and run demo
    try:
        demo = CognitionDemo(field_size=100, device=device)
        demo.run_interactive_menu()
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
