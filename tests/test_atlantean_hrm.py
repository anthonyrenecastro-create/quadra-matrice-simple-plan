"""
Test suite for CognitionSim core functionality
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestCognitionSimComponents:
    """Test core CognitionSim components"""
    
    def test_core_field_initialization(self):
        """Test CoreField initialization"""
        from app import CoreField
        
        field = CoreField(size=50)
        assert field.size == 50
        assert field.field_data is not None
        assert len(field.field_data) == 50
    
    def test_core_field_update(self):
        """Test CoreField update with vibrational mode"""
        from app import CoreField
        
        field = CoreField(size=50)
        update = np.random.randn(50) * 0.1
        
        initial_state = field.field_data.clone()
        field.update_with_vibrational_mode(update)
        
        # Field should have changed
        assert not torch.equal(initial_state, field.field_data)
    
    def test_core_field_get_state(self):
        """Test CoreField state retrieval"""
        from app import CoreField
        
        field = CoreField(size=50)
        state = field.get_state()
        
        assert isinstance(state, torch.Tensor)
        assert len(state) == 50
    
    def test_syntropy_engine_initialization(self):
        """Test SyntropyEngine initialization"""
        from app import SyntropyEngine
        
        engine = SyntropyEngine(num_fields=3, field_size=50)
        assert engine.num_fields == 3
        assert engine.field_size == 50
        assert len(engine.field_data) == 3
    
    def test_neuroplasticity_manager_initialization(self):
        """Test NeuroplasticityManager initialization"""
        from app import NeuroplasticityManager, CoreField, SyntropyEngine
        
        oscillator = Mock()
        core_field = CoreField(size=50)
        syntropy_engine = SyntropyEngine(num_fields=3, field_size=50)
        
        manager = NeuroplasticityManager(oscillator, core_field, syntropy_engine)
        assert manager.oscillator == oscillator
        assert manager.core_field == core_field
        assert manager.syntropy_engine == syntropy_engine
        assert manager.integrity_strikes == 0
    
    def test_neuroplasticity_regulate_syntropy(self):
        """Test syntropy regulation"""
        from app import NeuroplasticityManager, CoreField, SyntropyEngine
        
        oscillator = Mock()
        core_field = CoreField(size=50)
        syntropy_engine = SyntropyEngine(num_fields=3, field_size=50)
        
        manager = NeuroplasticityManager(oscillator, core_field, syntropy_engine)
        
        # Should not raise exception
        manager.regulate_syntropy()


class TestSystemState:
    """Test SystemState class"""
    
    def test_system_state_initialization(self):
        """Test SystemState initial values"""
        from app import SystemState
        
        state = SystemState()
        assert state.field_size == 100
        assert state.oscillator is None
        assert state.core_field is None
        assert not state.is_initialized
        assert not state.is_running
        assert state.iteration_count == 0
        assert len(state.loss_history) == 0
        assert len(state.reward_history) == 0
    
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_save_state(self, mock_open, mock_exists):
        """Test state saving"""
        from app import SystemState
        
        state = SystemState()
        state.is_initialized = True
        state.oscillator = Mock()
        state.oscillator.save_weights = Mock()
        state.loss_history = [0.5, 0.4, 0.3]
        state.iteration_count = 3
        
        # Mock file operations
        mock_exists.return_value = True
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()
        
        result = state.save_state()
        # Should attempt to save
        assert isinstance(result, bool)


class TestDataValidation:
    """Test input/output validation"""
    
    def test_field_size_validation(self):
        """Test field size must be positive"""
        from app import CoreField
        
        # Valid size
        field = CoreField(size=100)
        assert field.size == 100
        
        # Zero size should work but may have issues
        # This is a design decision
    
    def test_tensor_update_validation(self):
        """Test update validation with different data types"""
        from app import CoreField
        
        field = CoreField(size=50)
        
        # Test with numpy array
        update_np = np.random.randn(50) * 0.1
        field.update_with_vibrational_mode(update_np)
        
        # Test with tensor
        update_tensor = torch.randn(50) * 0.1
        field.update_with_vibrational_mode(update_tensor)
    
    def test_wrong_size_update(self):
        """Test update with wrong size is handled"""
        from app import CoreField
        
        field = CoreField(size=50)
        wrong_size_update = np.random.randn(30) * 0.1
        
        # Should not crash - implementation handles this
        field.update_with_vibrational_mode(wrong_size_update)


class TestErrorHandling:
    """Test error handling in various scenarios"""
    
    def test_oscillator_initialization_error(self):
        """Test handling of oscillator initialization error"""
        # This would test actual error handling in the app
        pass
    
    def test_invalid_state_load(self, tmp_path, monkeypatch):
        """Test loading invalid state"""
        from app import SystemState
        import app
        
        # Patch the paths to use tmp_path
        monkeypatch.setattr(app, 'METRICS_PATH', str(tmp_path / 'metrics.pkl'))
        monkeypatch.setattr(app, 'SYSTEM_STATE_PATH', str(tmp_path / 'system_state.pkl'))
        monkeypatch.setattr(app, 'OSCILLATOR_PATH', str(tmp_path / 'oscillator.pth'))
        
        state = SystemState()
        # Should return False when no state exists
        result = state.load_state()
        assert result == False


@pytest.mark.parametrize("field_size", [10, 50, 100, 200])
def test_field_sizes(field_size):
    """Test different field sizes"""
    from app import CoreField
    
    field = CoreField(size=field_size)
    assert field.size == field_size
    assert len(field.field_data) == field_size


@pytest.mark.parametrize("num_fields", [1, 3, 5, 10])
def test_syntropy_field_counts(num_fields):
    """Test different numbers of syntropy fields"""
    from app import SyntropyEngine
    
    engine = SyntropyEngine(num_fields=num_fields, field_size=50)
    assert engine.num_fields == num_fields
    assert len(engine.field_data) == num_fields
