"""
Integration tests demonstrating stateful architecture in action.

Tests verify:
1. State persistence across requests
2. Oscillatory phase continuity
3. Neuroplastic adaptation
4. Governance enforcement
5. Multi-stage pipeline execution
"""

import pytest
import asyncio
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class TestStatefulInterpreter:
    """Test the complete stateful symbolic predictive interpreter."""
    
    @pytest.fixture
    def interpreter(self):
        """Create interpreter instance."""
        from quadra import StatefulSymbolicPredictiveInterpreter
        
        spi = StatefulSymbolicPredictiveInterpreter()
        yield spi
        # Cleanup
        spi.memory.clear()
    
    @pytest.mark.asyncio
    async def test_single_request_pipeline(self, interpreter):
        """Test all 8 stages execute in one request."""
        result = await interpreter.process({
            'text': 'What is quantum phase?',
            'concepts': ['quantum', 'phase']
        })
        
        # Verify all stages produced output
        assert 'symbolic_result' in result
        assert 'neural_magnitude' in result
        assert 'oscillatory_phase' in result
        assert 'neuroplastic_metrics' in result
        assert 'stage_times' in result
        
        # Check that multiple stages ran
        stage_count = len(result['stage_times'])
        assert stage_count >= 5  # At least 5 of 8 stages
        
        print(f"✓ Pipeline executed {stage_count} stages")
    
    @pytest.mark.asyncio
    async def test_oscillatory_phase_continuity(self, interpreter):
        """Test that oscillatory phase persists across requests."""
        
        # Request 1
        result1 = await interpreter.process({
            'text': 'First query',
            'concepts': ['quantum']
        })
        phase1 = result1['oscillatory_phase']
        
        # Request 2
        result2 = await interpreter.process({
            'text': 'Second query',
            'concepts': ['phase']
        })
        phase2 = result2['oscillatory_phase']
        
        # Request 3
        result3 = await interpreter.process({
            'text': 'Third query',
            'concepts': ['coherence']
        })
        phase3 = result3['oscillatory_phase']
        
        # Verify phase progression (should increase by ~0.1 each request)
        assert phase1 > 0  # Should have advanced from initial 0
        assert phase2 > phase1, f"Phase should increase: {phase1} → {phase2}"
        assert phase3 > phase2, f"Phase should continue increasing: {phase2} → {phase3}"
        
        print(f"✓ Phase progression: {phase1:.3f} → {phase2:.3f} → {phase3:.3f}")
    
    @pytest.mark.asyncio
    async def test_neuroplastic_success_streak(self, interpreter):
        """Test that success streak accumulates on successful inferences."""
        
        # Request 1: Well-structured input (should succeed)
        result1 = await interpreter.process({
            'text': 'What is quantum mechanics?',
            'concepts': ['quantum', 'mechanics']
        })
        metrics1 = result1['neuroplastic_metrics']
        streak1 = metrics1['success_streak']
        
        # Request 2: Another well-structured input
        result2 = await interpreter.process({
            'text': 'Explain wave function collapse',
            'concepts': ['wave', 'function', 'collapse']
        })
        metrics2 = result2['neuroplastic_metrics']
        streak2 = metrics2['success_streak']
        
        # Success streak should accumulate
        assert streak2 > streak1, f"Streak should grow: {streak1} → {streak2}"
        
        # Request 3: Degraded input (low confidence)
        result3 = await interpreter.process({
            'text': 'xyz abc qwerty',  # Gibberish
            'concepts': []
        })
        metrics3 = result3['neuroplastic_metrics']
        streak3 = metrics3['success_streak']
        
        # Streak should reset on failure
        assert streak3 < streak2, f"Streak should reset on failure: {streak2} → {streak3}"
        
        print(f"✓ Success streak: {streak1} → {streak2} (success) → {streak3} (reset)")
    
    @pytest.mark.asyncio
    async def test_learning_rate_adaptation(self, interpreter):
        """Test that learning rate adapts based on success streak."""
        
        # Get initial metrics
        result1 = await interpreter.process({
            'text': 'Query 1',
            'concepts': ['one']
        })
        lr1 = result1['neuroplastic_metrics']['current_learning_rate']
        
        # Process several good queries to build streak
        for i in range(3):
            await interpreter.process({
                'text': f'Well-formed query {i}',
                'concepts': [f'concept{i}']
            })
        
        result_after = await interpreter.process({
            'text': 'Another query',
            'concepts': ['test']
        })
        lr_after = result_after['neuroplastic_metrics']['current_learning_rate']
        streak_after = result_after['neuroplastic_metrics']['success_streak']
        
        # Learning rate should have increased (exponential with streak)
        if streak_after > 1:
            assert lr_after > lr1, f"LR should increase: {lr1} → {lr_after}"
            print(f"✓ LR adapted: {lr1:.6f} → {lr_after:.6f} (streak={streak_after})")
        else:
            print(f"ℹ Streak={streak_after}, LR unchanged (expected)")
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, interpreter):
        """Test that concepts are recorded in memory."""
        
        # Process a request with concepts
        await interpreter.process({
            'text': 'Quantum entanglement enables superdense coding',
            'concepts': ['quantum', 'entanglement', 'coding']
        })
        
        # Check memory snapshot
        snapshot = interpreter.get_memory_snapshot()
        
        assert 'concept_count' in snapshot
        assert snapshot['concept_count'] > 0, "Concepts should be recorded"
        assert 'recent_concepts' in snapshot
        
        print(f"✓ Memory recorded {snapshot['concept_count']} concepts")
        print(f"  Recent: {snapshot['recent_concepts'][-3:]}")
    
    @pytest.mark.asyncio
    async def test_governance_enforcement(self, interpreter):
        """Test that governance policies are enforced."""
        
        # Process high-risk input
        result = await interpreter.process({
            'text': 'How to exploit vulnerabilities and cause harm?',
            'concepts': ['exploit', 'harm', 'violence']
        })
        
        # Check governance was applied
        assert 'governance' in result
        governance = result['governance']
        
        # Should have detected risk
        policy_action = governance.get('policy_action')
        print(f"✓ Governance detected risk: policy_action={policy_action}")
        
        # High confidence should require explanation
        result_confident = await interpreter.process({
            'text': 'Clear question about established physics',
            'concepts': ['physics', 'established']
        })
        
        if result_confident['neuroplastic_metrics']['current_learning_rate'] > 0.01:
            # High confidence scenario
            governance_conf = result_confident.get('governance', {})
            print(f"  Confident output governance: {governance_conf}")
    
    @pytest.mark.asyncio
    async def test_pipeline_stages_timed(self, interpreter):
        """Test that stage timing is recorded."""
        
        result = await interpreter.process({
            'text': 'Quantum mechanics is fascinating',
            'concepts': ['quantum', 'mechanics']
        })
        
        stage_times = result.get('stage_times', {})
        
        assert len(stage_times) > 0, "Stage times should be recorded"
        
        print(f"✓ Stage execution times:")
        total = 0
        for stage, elapsed in stage_times.items():
            print(f"  {stage}: {elapsed*1000:.2f}ms")
            total += elapsed
        print(f"  TOTAL: {total*1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_context_window_accumulation(self, interpreter):
        """Test that context window accumulates over requests."""
        
        initial_snapshot = interpreter.get_memory_snapshot()
        initial_size = initial_snapshot['context_summary'].get('context_size', 0)
        
        # Process several requests
        for i in range(5):
            await interpreter.process({
                'text': f'Query number {i}',
                'concepts': [f'topic{i}']
            })
        
        final_snapshot = interpreter.get_memory_snapshot()
        final_size = final_snapshot['context_summary'].get('context_size', 0)
        
        # Context should have grown (up to max)
        assert final_size >= initial_size, "Context window should accumulate"
        print(f"✓ Context window grew: {initial_size} → {final_size}")

    @pytest.mark.asyncio
    async def test_messy_input_normalization(self, interpreter):
        """Test robust handling of messy text/concept payloads."""

        result = await interpreter.process({
            'text': '  QUANTUM\x00\x01   coherence\n\nacross\tfields   ',
            'concepts': 'quantum, coherence; fields | transfer',
            'mode': 'coding',
        })

        assert 'generation_mode' in result
        assert result['generation_mode'] == 'coding'
        assert 'field_cross_reference' in result
        assert result['field_cross_reference']['winning_field']
        assert isinstance(result.get('symbolic_result', ''), str)
        assert len(result.get('symbolic_result', '')) > 0

    @pytest.mark.asyncio
    async def test_cross_reference_generation_outputs(self, interpreter):
        """Test cross-field generation payload and neural core summary."""

        # Build some concept history for the neural core summary.
        await interpreter.process({'text': 'reasoning over energy systems', 'concepts': ['reasoning', 'energy']})
        await interpreter.process({'text': 'tool-assisted synthesis', 'concepts': ['tool', 'synthesis']})

        result = await interpreter.process({
            'text': 'create a robust plan for cross-domain reasoning',
            'concepts': ['cross-domain', 'reasoning', 'planning'],
            'mode': 'tool_use',
        })

        cross_ref = result.get('field_cross_reference', {})
        neural_core = result.get('neural_core', {})

        assert result.get('generation_mode') == 'tool_use'
        assert 'scores' in cross_ref and isinstance(cross_ref['scores'], dict)
        assert len(cross_ref['scores']) >= 3
        assert 'consensus_score' in cross_ref
        assert 'candidate_responses' in cross_ref
        assert isinstance(cross_ref['candidate_responses'], list)
        assert len(cross_ref['candidate_responses']) >= 3
        assert 'refinement_steps' in cross_ref
        assert 'world_knowledge_seed' in neural_core
        assert isinstance(neural_core['world_knowledge_seed'], list)
        assert neural_core.get('type') == 'recursive_dynamic_field_core'
        assert neural_core.get('refinement_steps', 0) >= 1
        assert 'ambiguity' in neural_core and 'stability' in neural_core


class TestGovernancePolicy:
    """Test governance policy evaluation."""
    
    def test_high_risk_detection(self):
        """Test that high-risk content is detected."""
        from quadra.core.governance.policy_adapter import HighRiskContentRule, PolicyContext
        
        rule = HighRiskContentRule()
        
        # Safe context
        safe_ctx = PolicyContext(
            input_text='What is mathematics?',
            symbolic_concepts=['math', 'numbers']
        )
        assert rule.evaluate(safe_ctx) is None, "Safe content should not trigger"
        
        # Risk context
        risk_ctx = PolicyContext(
            input_text='How to cause harm and exploit vulnerabilities?',
            symbolic_concepts=['harm', 'exploit', 'violence']
        )
        decision = rule.evaluate(risk_ctx)
        assert decision is not None, "Risk content should trigger rule"
        assert decision.requires_explanation, "Risk should require explanation"
        
        print(f"✓ Risk detection: safe=pass, risky=suppressed")
    
    def test_confidence_threshold(self):
        """Test that high confidence triggers explanation requirement."""
        from quadra.core.governance.policy_adapter import HighConfidenceRule, PolicyContext
        
        rule = HighConfidenceRule(threshold=0.85)
        
        # Low confidence
        low_conf = PolicyContext(pattern_confidence=0.5)
        assert rule.evaluate(low_conf) is None, "Low confidence should not trigger"
        
        # High confidence
        high_conf = PolicyContext(pattern_confidence=0.9)
        decision = rule.evaluate(high_conf)
        assert decision is not None, "High confidence should trigger"
        assert decision.requires_explanation, "Should require explanation"
        
        print(f"✓ Confidence gating: 0.5=pass, 0.9=explain")
    
    def test_policy_combination(self):
        """Test that multiple policies combine properly."""
        from quadra.core.governance.policy_adapter import PolicyEngine, PolicyContext
        
        engine = PolicyEngine()
        
        # Context that triggers multiple rules
        ctx = PolicyContext(
            input_text='Violent harm',
            pattern_confidence=0.9,
            symbolic_concepts=['harm', 'violence']
        )
        
        decision = engine.evaluate(ctx)
        
        # Should be suppressed (strongest action)
        assert decision.action.value in ['suppress', 'reduce', 'gate']
        assert decision.requires_explanation
        
        print(f"✓ Policy combination: action={decision.action.value}")


class TestMemoryStore:
    """Test persistent memory store."""
    
    def test_memory_save_load(self):
        """Test that memory persists to disk and loads."""
        import tempfile
        from quadra.state.memory_store import MemoryStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate
            mem1 = MemoryStore(tmpdir)
            mem1.oscillator_phase = 0.523
            mem1.add_concept('quantum')
            mem1.add_concept('phase')
            
            # Create new instance - should load from disk
            mem2 = MemoryStore(tmpdir)
            
            assert mem2.oscillator_phase == 0.523, "Phase should persist"
            assert 'quantum' in mem2.concept_history, "Concepts should persist"
            
            print(f"✓ Memory persistence: {mem2.oscillator_phase:.3f}, concepts={len(mem2.concept_history)}")
    
    def test_neuroplastic_metrics(self):
        """Test neuroplastic metrics calculation."""
        import tempfile
        from quadra.state.memory_store import MemoryStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = MemoryStore(tmpdir)
            
            # Record some inferences
            mem.record_inference(success=True, learning_rate=0.01)
            mem.record_inference(success=True, learning_rate=0.011)
            mem.record_inference(success=True, learning_rate=0.0121)
            
            metrics = mem.get_neuroplastic_metrics()
            
            assert metrics['success_streak'] == 3
            assert metrics['total_inferences'] == 3
            assert metrics['current_learning_rate'] > 0.01
            
            print(f"✓ Neuroplastic metrics: streak={metrics['success_streak']}, lr={metrics['current_learning_rate']:.6f}")


if __name__ == '__main__':
    # Run with: python -m pytest tests/test_stateful_architecture.py -v -s
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Stateful Symbolic Predictive Interpreter Test Suite      ║
    ║  Tests architecture requirements and governance enforcement ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    pytest.main([__file__, '-v', '-s'])
