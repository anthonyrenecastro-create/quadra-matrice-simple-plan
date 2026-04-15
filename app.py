from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import torch
import numpy as np
from datetime import datetime
import json
import threading
import time
import os
import pickle
import logging
import traceback

# Import configuration
try:
    from config import get_config
    config = get_config()
except ImportError:
    # Fallback if config not available
    class config:
        HOST = '0.0.0.0'
        PORT = 5000
        DASHBOARD_STATE_DIR = 'dashboard_state'
        LOG_LEVEL = 'INFO'
        LOGS_DIR = 'logs'

# Import utilities
try:
    from utils import (
        handle_errors,
        handle_api_errors,
        safe_execute,
        ErrorContext,
        InitializationError,
        TrainingError,
        StateError,
        validate_field_size,
        validate_text_input,
        setup_logging,
        log_performance
    )
    from utils.model_versioning import ModelVersionManager, generate_version_string
    from utils.model_monitoring import ModelMonitor, HealthChecker
    from utils.model_modes import ModelMode, mode_manager, require_mode, inference_only, training_only
    from utils.security import (
        secrets_manager,
        rate_limiter,
        init_auth,
        require_auth,
        require_permission,
        rate_limit
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    UTILS_AVAILABLE = False
    print(f"WARNING: Utils module not available, using basic error handling: {e}")

# Import the main components
from atlantean_hrm_spi import (
    OscillatorySynapseTheory,
    PatternModule,
    SymbolicConfig,
    SymbolicPredictiveInterpreter
)

# Import noise injection module
try:
    from utils.noise_injection import NoiseInjector, NoiseType
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("WARNING: Noise injection module not available")

# Import ablation study models
try:
    from utils.ablation_models import AblationConfig, SimpleMLPBaseline, SimpleQLearner, MLPQLearner
    ABLATION_AVAILABLE = True
except ImportError:
    ABLATION_AVAILABLE = False
    print("WARNING: Ablation study models not available")

# Setup advanced logging if available
if UTILS_AVAILABLE:
    try:
        log_dir = getattr(config, 'LOGS_DIR', 'logs')
        setup_logging(
            log_level=getattr(config, 'LOG_LEVEL', 'INFO'),
            log_dir=log_dir,
            app_name='atlantean_hrm',
            enable_console=True,
            enable_file=True
        )
    except Exception as e:
        logging.basicConfig(
            level=getattr(logging, getattr(config, 'LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
else:
    logging.basicConfig(
        level=getattr(logging, getattr(config, 'LOG_LEVEL', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Create simple wrapper classes for missing components
class CoreField:
    def __init__(self, size=100):
        self.size = size
        self.field_data = torch.randn(size) * 0.1
    
    def update_with_vibrational_mode(self, update):
        if isinstance(update, np.ndarray):
            update = torch.tensor(update, dtype=torch.float32)
        if len(update) == len(self.field_data):
            self.field_data = self.field_data * 0.9 + update * 0.1
        
    def get_state(self):
        return self.field_data

class SyntropyEngine:
    def __init__(self, num_fields=3, field_size=100):
        self.num_fields = num_fields
        self.field_size = field_size
        self.field_data = [np.random.randn(field_size) * 0.1 for _ in range(num_fields)]

class NeuroplasticityManager:
    def __init__(self, oscillator, core_field, syntropy_engine):
        self.oscillator = oscillator
        self.core_field = core_field
        self.syntropy_engine = syntropy_engine
        self.integrity_strikes = 0
        
    def regulate_syntropy(self):
        for i in range(self.syntropy_engine.num_fields):
            field_mean = np.mean(self.syntropy_engine.field_data[i])
            if abs(field_mean - 0.5) > 0.1:
                self.syntropy_engine.field_data[i] += (0.5 - field_mean) * 0.1
                self.syntropy_engine.field_data[i] = np.clip(self.syntropy_engine.field_data[i], -1, 1)

app = Flask(__name__)
app.config['SECRET_KEY'] = getattr(config, 'SECRET_KEY', 'quadramatrix_secret_key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=False, ping_timeout=60, ping_interval=25)

# Persistent storage paths
STATE_DIR = getattr(config, 'DASHBOARD_STATE_DIR', 'dashboard_state')
if isinstance(STATE_DIR, str):
    os.makedirs(STATE_DIR, exist_ok=True)
else:
    STATE_DIR = str(STATE_DIR)
    os.makedirs(STATE_DIR, exist_ok=True)

OSCILLATOR_PATH = os.path.join(STATE_DIR, 'oscillator_weights.pth')
METRICS_PATH = os.path.join(STATE_DIR, 'metrics_history.pkl')
SYSTEM_STATE_PATH = os.path.join(STATE_DIR, 'system_state.pkl')

# Initialize ML-specific components
version_manager = None
model_monitor = None
health_checker = None

if UTILS_AVAILABLE:
    try:
        from pathlib import Path
        models_dir = Path(STATE_DIR) / "models"
        monitoring_dir = Path(STATE_DIR) / "monitoring"
        
        version_manager = ModelVersionManager(models_dir)
        model_monitor = ModelMonitor(monitoring_dir)
        health_checker = HealthChecker(
            model_path=models_dir,
            required_files=["model_registry.json"]
        )
        logger.info("ML-specific components initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize ML components: {e}")

# Global system state
class SystemState:
    def __init__(self):
        self.field_size = 100
        self.oscillator = None
        self.core_field = None
        self.syntropy_engine = None
        self.pattern_module = None
        self.neuroplasticity_manager = None
        self.symbolic_interpreter = None
        
        # Noise injection
        self.noise_injector = None
        self.noise_enabled = False
        self.noise_intensity = 0.15
        self.noise_stats = {'total_injections': 0, 'by_type': {}}
        
        # Ablation study configuration
        self.ablation_config = AblationConfig() if ABLATION_AVAILABLE else None
        self.baseline_model = None
        
        self.loss_history = []
        self.reward_history = []
        self.variance_history = []
        self.field_mean_history = []
        self.iteration_count = 0
        self.is_running = False
        self.is_initialized = False
        
        self.training_thread = None
        self.current_version = None
    
    def save_state(self):
        """Save system state to disk"""
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            
            # Save oscillator weights
            if self.oscillator is not None:
                self.oscillator.save_weights(OSCILLATOR_PATH)
            
            # Save metrics history
            metrics = {
                'loss_history': self.loss_history,
                'reward_history': self.reward_history,
                'variance_history': self.variance_history,
                'field_mean_history': self.field_mean_history,
                'iteration_count': self.iteration_count
            }
            with open(METRICS_PATH, 'wb') as f:
                pickle.dump(metrics, f)
            
            # Save system flags
            system_info = {
                'field_size': self.field_size,
                'is_initialized': self.is_initialized,
                'integrity_strikes': self.neuroplasticity_manager.integrity_strikes if self.neuroplasticity_manager else 0
            }
            with open(SYSTEM_STATE_PATH, 'wb') as f:
                pickle.dump(system_info, f)
            
            print(f"✓ State saved: {self.iteration_count} iterations")
            return True
        except Exception as e:
            print(f"✗ Failed to save state: {e}")
            return False
    
    def load_state(self):
        """Load system state from disk"""
        try:
            # Check if state files exist
            if not os.path.exists(METRICS_PATH):
                print("No saved state found")
                return False
            
            # Load system info first
            with open(SYSTEM_STATE_PATH, 'rb') as f:
                system_info = pickle.load(f)
            
            self.field_size = system_info.get('field_size', 100)
            
            # Initialize components
            self.oscillator = OscillatorySynapseTheory(field_size=self.field_size)
            self.core_field = CoreField(size=self.field_size)
            self.syntropy_engine = SyntropyEngine(num_fields=3, field_size=self.field_size)
            self.pattern_module = PatternModule(n_clusters=3)
            self.neuroplasticity_manager = NeuroplasticityManager(
                self.oscillator, self.core_field, self.syntropy_engine
            )
            symbolic_config = SymbolicConfig()
            self.symbolic_interpreter = SymbolicPredictiveInterpreter(
                self.pattern_module, self.core_field, symbolic_config
            )
            
            # Load oscillator weights
            if os.path.exists(OSCILLATOR_PATH):
                self.oscillator.load_weights(OSCILLATOR_PATH)
            
            # Load metrics
            with open(METRICS_PATH, 'rb') as f:
                metrics = pickle.load(f)
            
            self.loss_history = metrics['loss_history']
            self.reward_history = metrics['reward_history']
            self.variance_history = metrics['variance_history']
            self.field_mean_history = metrics['field_mean_history']
            self.iteration_count = metrics['iteration_count']
            
            # Restore integrity strikes
            self.neuroplasticity_manager.integrity_strikes = system_info.get('integrity_strikes', 0)
            
            self.is_initialized = True
            print(f"✓ State loaded: {self.iteration_count} iterations restored")
            return True
        except Exception as e:
            print(f"✗ Failed to load state: {e}")
            return False

state = SystemState()

def emit_update(event, data):
    """Helper to emit updates to all clients"""
    socketio.emit(event, data)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/health')
def health():
    """Health check endpoint for Docker/K8s"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': getattr(config, 'VERSION', '1.0.0'),
            'initialized': state.is_initialized,
            'running': state.is_running
        }
        return jsonify(health_status), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'initialized': state.is_initialized,
        'running': state.is_running,
        'iteration_count': state.iteration_count,
        'noise': {
            'enabled': state.noise_enabled,
            'intensity': state.noise_intensity,
            'available': NOISE_AVAILABLE,
            'stats': state.noise_stats
        },
        'current_metrics': {
            'loss': state.loss_history[-1] if state.loss_history else None,
            'reward': state.reward_history[-1] if state.reward_history else None,
            'variance': state.variance_history[-1] if state.variance_history else None,
            'mean': state.field_mean_history[-1] if state.field_mean_history else None,
            'integrity_strikes': state.neuroplasticity_manager.integrity_strikes if state.neuroplasticity_manager else 0,
            'qtable_size': len(state.oscillator.q_table) if state.oscillator else 0
        },
        'history': {
            'loss': state.loss_history[-50:],
            'reward': state.reward_history[-50:],
            'variance': state.variance_history[-50:],
            'mean': state.field_mean_history[-50:]
        }
    })

@app.route('/api/noise/toggle', methods=['POST'])
def api_toggle_noise():
    """Toggle noise injection via HTTP"""
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', not state.noise_enabled)
        state.noise_enabled = enabled
        
        if enabled and NOISE_AVAILABLE:
            if state.noise_injector is None:
                state.noise_injector = NoiseInjector(enabled=True, intensity=state.noise_intensity)
            else:
                state.noise_injector.set_enabled(True)
        elif state.noise_injector:
            state.noise_injector.set_enabled(False)
        
        return jsonify({
            'success': True,
            'noise_enabled': state.noise_enabled,
            'noise_intensity': state.noise_intensity,
            'available': NOISE_AVAILABLE
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/noise/intensity', methods=['POST'])
def api_set_noise_intensity():
    """Set noise intensity via HTTP"""
    try:
        data = request.get_json() or {}
        intensity = float(data.get('intensity', 0.15))
        intensity = max(0.0, min(1.0, intensity))
        state.noise_intensity = intensity
        
        if state.noise_injector is not None:
            state.noise_injector.set_intensity(intensity)
        
        return jsonify({
            'success': True,
            'noise_intensity': state.noise_intensity
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/noise/stats', methods=['GET'])
def api_get_noise_stats():
    """Get noise statistics via HTTP"""
    return jsonify({
        'enabled': state.noise_enabled,
        'intensity': state.noise_intensity,
        'available': NOISE_AVAILABLE,
        'stats': state.noise_stats
    })

@app.route('/api/ablation/config', methods=['GET', 'POST'])
def api_ablation_config():
    """Get or set ablation study configuration"""
    if request.method == 'GET':
        if state.ablation_config:
            return jsonify(state.ablation_config.to_dict())
        else:
            return jsonify({
                'noise_enabled': state.noise_enabled,
                'plasticity_enabled': True,
                'field_feedback_enabled': True,
                'model_type': 'full',
                'available': ABLATION_AVAILABLE
            })
    else:  # POST
        if not ABLATION_AVAILABLE or state.ablation_config is None:
            return jsonify({'error': 'Ablation studies not available'}), 503
        
        data = request.json
        state.ablation_config.from_dict(data)
        return jsonify({
            'success': True,
            'config': state.ablation_config.to_dict(),
            'description': state.ablation_config.get_description()
        })

@app.route('/api/ablation/compare', methods=['POST'])
def api_ablation_compare():
    """Run multiple ablation configurations and compare results"""
    if not ABLATION_AVAILABLE:
        return jsonify({'error': 'Ablation studies not available'}), 503
    
    configs = request.json.get('configs', [])
    results = []
    
    for config_dict in configs:
        # Save current state
        original_config = state.ablation_config.to_dict() if state.ablation_config else {}
        
        # Apply config
        if state.ablation_config:
            state.ablation_config.from_dict(config_dict)
        
        # Collect metrics (would need to run actual training - simplified here)
        result = {
            'config': config_dict,
            'description': state.ablation_config.get_description() if state.ablation_config else 'N/A',
            'metrics': {
                'avg_loss': np.mean(state.loss_history[-10:]) if len(state.loss_history) >= 10 else 0,
                'avg_reward': np.mean(state.reward_history[-10:]) if len(state.reward_history) >= 10 else 0,
                'avg_variance': np.mean(state.variance_history[-10:]) if len(state.variance_history) >= 10 else 0,
                'iterations': state.iteration_count
            }
        }
        results.append(result)
        
        # Restore original config
        if state.ablation_config:
            state.ablation_config.from_dict(original_config)
    
    return jsonify({'results': results, 'count': len(results)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    socketio.emit('status_message', {'message': 'Connected to server', 'type': 'info'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('initialize_system')
def handle_initialize():
    """Initialize the CognitionSim system"""
    try:
        socketio.emit('status_message', {'message': 'Initializing CognitionSim System...', 'type': 'info'})
        
        state.oscillator = OscillatorySynapseTheory(field_size=state.field_size)
        state.core_field = CoreField(size=state.field_size)
        state.syntropy_engine = SyntropyEngine(num_fields=3, field_size=state.field_size)
        state.pattern_module = PatternModule(n_clusters=3)
        state.neuroplasticity_manager = NeuroplasticityManager(
            state.oscillator, state.core_field, state.syntropy_engine
        )
        symbolic_config = SymbolicConfig()
        state.symbolic_interpreter = SymbolicPredictiveInterpreter(
            state.pattern_module, state.core_field, symbolic_config
        )
        
        state.is_initialized = True
        socketio.emit('status_message', {'message': 'System initialized successfully!', 'type': 'success'})
        socketio.emit('system_initialized', {'initialized': True})
        
    except Exception as e:
        socketio.emit('status_message', {'message': f'Initialization failed: {str(e)}', 'type': 'error'})

@socketio.on('start_training')
def handle_start_training():
    """Start the training process"""
    if not state.is_initialized:
        socketio.emit('status_message', {'message': 'System not initialized!', 'type': 'error'})
        return
    
    if state.is_running:
        socketio.emit('status_message', {'message': 'Training already running!', 'type': 'warning'})
        return
    
    state.is_running = True
    socketio.emit('status_message', {'message': 'Starting training...', 'type': 'info'})
    socketio.emit('training_status', {'running': True})
    
    # Start training in a separate thread
    state.training_thread = threading.Thread(target=training_loop, daemon=True)
    state.training_thread.start()

@socketio.on('stop_training')
def handle_stop_training():
    """Stop the training process"""
    state.is_running = False
    state.save_state()  # Save when stopping
    socketio.emit('status_message', {'message': 'Training stopped. State saved.', 'type': 'info'})
    socketio.emit('training_status', {'running': False})

@socketio.on('load_state')
def handle_load_state():
    """Load previously saved state"""
    try:
        if state.load_state():
            socketio.emit('status_message', {'message': f'State loaded! {state.iteration_count} iterations restored.', 'type': 'success'})
            socketio.emit('system_initialized', {'initialized': True})
            # Send history data
            socketio.emit('state_loaded', {
                'iteration_count': state.iteration_count,
                'history': {
                    'loss': state.loss_history[-50:],
                    'reward': state.reward_history[-50:],
                    'variance': state.variance_history[-50:],
                    'mean': state.field_mean_history[-50:]
                }
            })
        else:
            socketio.emit('status_message', {'message': 'No saved state found.', 'type': 'warning'})
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to load state: {str(e)}', 'type': 'error'})

@socketio.on('reset_system')
def handle_reset():
    """Reset the system"""
    state.is_running = False
    state.loss_history = []
    state.reward_history = []
    state.variance_history = []
    state.field_mean_history = []
    state.iteration_count = 0
    state.oscillator = None
    state.core_field = None
    state.is_initialized = False
    state.noise_stats = {'total_injections': 0, 'by_type': {}}
    
    # Delete saved state files
    try:
        if os.path.exists(STATE_DIR):
            for file in [OSCILLATOR_PATH, METRICS_PATH, SYSTEM_STATE_PATH]:
                if os.path.exists(file):
                    os.remove(file)
            socketio.emit('status_message', {'message': 'System reset. Saved state deleted.', 'type': 'info'})
    except Exception as e:
        socketio.emit('status_message', {'message': f'System reset. Warning: {str(e)}', 'type': 'warning'})
    
    socketio.emit('system_reset', {})

@socketio.on('toggle_noise')
def handle_toggle_noise(data):
    """Toggle noise injection on/off"""
    try:
        enabled = data.get('enabled', False)
        state.noise_enabled = enabled
        
        if enabled and NOISE_AVAILABLE:
            if state.noise_injector is None:
                state.noise_injector = NoiseInjector(enabled=True, intensity=state.noise_intensity)
            else:
                state.noise_injector.set_enabled(True)
            message = f'Noise injection ENABLED (intensity={state.noise_intensity})'
        else:
            if state.noise_injector:
                state.noise_injector.set_enabled(False)
            message = 'Noise injection DISABLED'
        
        socketio.emit('status_message', {'message': message, 'type': 'success'})
        socketio.emit('noise_status', {
            'enabled': state.noise_enabled,
            'intensity': state.noise_intensity,
            'available': NOISE_AVAILABLE
        })
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to toggle noise: {str(e)}', 'type': 'error'})

@socketio.on('set_noise_intensity')
def handle_set_noise_intensity(data):
    """Set noise injection intensity"""
    try:
        intensity = float(data.get('intensity', 0.15))
        intensity = max(0.0, min(1.0, intensity))  # Clamp to [0, 1]
        state.noise_intensity = intensity
        
        if state.noise_injector is not None:
            state.noise_injector.set_intensity(intensity)
        
        socketio.emit('status_message', {
            'message': f'Noise intensity set to {intensity:.2f}',
            'type': 'success'
        })
        socketio.emit('noise_status', {
            'enabled': state.noise_enabled,
            'intensity': state.noise_intensity,
            'available': NOISE_AVAILABLE
        })
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to set intensity: {str(e)}', 'type': 'error'})

@socketio.on('get_noise_stats')
def handle_get_noise_stats():
    """Get noise injection statistics"""
    try:
        stats = state.noise_stats.copy() if state.noise_stats else {'total_injections': 0, 'by_type': {}}
        socketio.emit('noise_stats', stats)
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to get stats: {str(e)}', 'type': 'error'})

# Ablation Study Handlers
@socketio.on('toggle_plasticity')
def handle_toggle_plasticity(data):
    """Toggle weight plasticity (frozen vs trainable)"""
    try:
        if not ABLATION_AVAILABLE or state.ablation_config is None:
            socketio.emit('status_message', {'message': 'Ablation studies not available', 'type': 'error'})
            return
        
        enabled = data.get('enabled', True)
        state.ablation_config.plasticity_enabled = enabled
        
        # Apply immediately if model exists
        if state.spi and hasattr(state.spi, 'snn'):
            for param in state.spi.snn.parameters():
                param.requires_grad = enabled
        
        message = f'Plasticity {"ENABLED" if enabled else "FROZEN"}'
        socketio.emit('status_message', {'message': message, 'type': 'success'})
        socketio.emit('ablation_status', state.ablation_config.to_dict())
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to toggle plasticity: {str(e)}', 'type': 'error'})

@socketio.on('toggle_field_feedback')
def handle_toggle_field_feedback(data):
    """Toggle quantum field feedback"""
    try:
        if not ABLATION_AVAILABLE or state.ablation_config is None:
            socketio.emit('status_message', {'message': 'Ablation studies not available', 'type': 'error'})
            return
        
        enabled = data.get('enabled', True)
        state.ablation_config.field_feedback_enabled = enabled
        
        message = f'Field Feedback {"ENABLED" if enabled else "DISABLED"}'
        socketio.emit('status_message', {'message': message, 'type': 'success'})
        socketio.emit('ablation_status', state.ablation_config.to_dict())
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to toggle field feedback: {str(e)}', 'type': 'error'})

@socketio.on('set_model_type')
def handle_set_model_type(data):
    """Set model type for ablation study"""
    try:
        if not ABLATION_AVAILABLE or state.ablation_config is None:
            socketio.emit('status_message', {'message': 'Ablation studies not available', 'type': 'error'})
            return
        
        model_type = data.get('model_type', 'full')
        state.ablation_config.model_type = model_type
        
        # Initialize baseline model if needed
        if model_type == 'mlp':
            state.baseline_model = SimpleMLPBaseline(input_dim=16, hidden_dim=128, output_dim=4)
        elif model_type == 'q_learner':
            state.baseline_model = SimpleQLearner(state_dim=16, action_dim=4)
        elif model_type == 'mlp_q':
            state.baseline_model = MLPQLearner(state_dim=16, action_dim=4, hidden_dim=64)
        else:
            state.baseline_model = None
        
        message = f'Model type set to: {model_type}'
        socketio.emit('status_message', {'message': message, 'type': 'success'})
        socketio.emit('ablation_status', state.ablation_config.to_dict())
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to set model type: {str(e)}', 'type': 'error'})

@socketio.on('get_ablation_status')
def handle_get_ablation_status():
    """Get current ablation configuration"""
    try:
        if ABLATION_AVAILABLE and state.ablation_config:
            socketio.emit('ablation_status', state.ablation_config.to_dict())
        else:
            socketio.emit('ablation_status', {
                'noise_enabled': state.noise_enabled,
                'plasticity_enabled': True,
                'field_feedback_enabled': True,
                'model_type': 'full'
            })
    except Exception as e:
        socketio.emit('status_message', {'message': f'Failed to get ablation status: {str(e)}', 'type': 'error'})

def training_loop():
    """Training loop running in separate thread"""
    test_texts = [
        "The quantum field oscillates with harmonic resonance",
        "Neural networks learn patterns through synaptic plasticity",
        "Syntropy represents order emerging from chaos",
        "Consciousness emerges from complex information processing",
        "Machine learning enables adaptive intelligence systems",
        "Spiking neurons communicate through temporal coding",
        "Deep learning architectures model hierarchical representations",
        "Reinforcement learning optimizes sequential decision making",
    ]
    
    try:
        while state.is_running:
            for text in test_texts:
                if not state.is_running:
                    break
                    
                state.iteration_count += 1
                
                # 🔬 Apply plasticity settings (freeze/unfreeze weights)
                if state.ablation_config is not None:
                    if not state.ablation_config.plasticity_enabled:
                        # Freeze all parameters
                        if hasattr(state, 'spi') and hasattr(state.spi, 'snn'):
                            for param in state.spi.snn.parameters():
                                param.requires_grad = False
                    else:
                        # Ensure parameters are trainable
                        if hasattr(state, 'spi') and hasattr(state.spi, 'snn'):
                            for param in state.spi.snn.parameters():
                                param.requires_grad = True
                
                # Run training
                feature_vector = state.oscillator.process_streamed_data(text)
                synthetic_data = state.oscillator.generate_synthetic_data(num_samples=10)
                
                # 🔊 Apply noise injection if enabled
                if state.noise_enabled and state.noise_injector is not None:
                    # Inject noise into synthetic data
                    noisy_synthetic_data = []
                    for data_point in synthetic_data:
                        noisy_point = state.noise_injector.inject(
                            data_point,
                            NoiseType.DROPOUT,
                            state.noise_intensity * 0.3
                        )
                        noisy_synthetic_data.append(noisy_point)
                    synthetic_data = noisy_synthetic_data
                    
                    # Inject noise into feature vector
                    feature_vector = state.noise_injector.inject(
                        feature_vector,
                        NoiseType.GAUSSIAN,
                        state.noise_intensity
                    )
                
                reward = state.oscillator.train(synthetic_data, text, epochs=3)
                
                # Update field
                field_update = feature_vector.cpu().numpy()
                
                # 🔬 Apply field feedback control
                if state.ablation_config is not None and not state.ablation_config.field_feedback_enabled:
                    # Disable field feedback - use random updates
                    field_update = np.random.randn(*field_update.shape) * 0.1
                
                state.core_field.update_with_vibrational_mode(field_update)
                
                # Calculate metrics
                field_state = state.core_field.get_state()
                
                # 🔊 Apply noise to field state if enabled
                if state.noise_enabled and state.noise_injector is not None:
                    field_state = state.noise_injector.inject(
                        field_state,
                        NoiseType.FIELD_PERTURBATION,
                        state.noise_intensity * 0.5
                    )
                
                field_var = torch.var(field_state).item()
                field_mean = torch.mean(field_state).item()
                
                # Get loss from last training
                with torch.no_grad():
                    test_field = state.oscillator.nn1.update_field(synthetic_data[0])
                    loss = (torch.var(test_field) + torch.abs(torch.mean(test_field) - 0.5)).item()
                
                # Store metrics
                state.loss_history.append(loss)
                state.reward_history.append(reward)
                state.variance_history.append(field_var)
                state.field_mean_history.append(field_mean)
                
                # Update noise stats
                if state.noise_enabled and state.noise_injector is not None:
                    state.noise_stats = state.noise_injector.get_stats()
                
                # Emit update to clients
                metrics = {
                    'iteration': state.iteration_count,
                    'loss': loss,
                    'reward': reward,
                    'variance': field_var,
                    'mean': field_mean,
                    'integrity_strikes': state.neuroplasticity_manager.integrity_strikes,
                    'qtable_size': len(state.oscillator.q_table),
                    'text': text,
                    'noise_enabled': state.noise_enabled,
                    'noise_intensity': state.noise_intensity,
                    'noise_injections': state.noise_stats.get('total_injections', 0),
                    'ablation_config': state.ablation_config.to_dict() if state.ablation_config else None
                }
                socketio.emit('metrics_update', metrics)
                
                # Regulate syntropy
                state.neuroplasticity_manager.regulate_syntropy()
                
                # Auto-save every 10 iterations
                if state.iteration_count % 10 == 0:
                    state.save_state()
                
                # Small delay
                time.sleep(0.5)
                
    except Exception as e:
        socketio.emit('status_message', {'message': f'Training error: {str(e)}', 'type': 'error'})
        state.is_running = False
        socketio.emit('training_status', {'running': False})


# ============================================================
# ML-SPECIFIC ENDPOINTS
# ============================================================

@app.route('/api/model/versions', methods=['GET'])
@rate_limit
def list_model_versions():
    """List all model versions"""
    if not version_manager:
        return jsonify({'error': 'Model versioning not available'}), 503
    
    try:
        versions = version_manager.list_versions()
        return jsonify({
            'versions': [
                {
                    'version': v.version,
                    'created_at': v.created_at,
                    'metrics': v.metrics,
                    'description': v.description,
                    'tags': v.tags,
                    'file_size': v.file_size
                }
                for v in versions
            ],
            'count': len(versions)
        })
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/save', methods=['POST'])
@rate_limit
@training_only
def save_model_version():
    """Save current model as new version"""
    if not version_manager or not state.oscillator:
        return jsonify({'error': 'Model or versioning not available'}), 503
    
    try:
        data = request.get_json() or {}
        description = data.get('description', 'Manual save')
        tags = data.get('tags', [])
        
        # Generate version
        version = generate_version_string(prefix="manual")
        
        # Calculate current metrics
        metrics = {
            'loss': float(np.mean(state.loss_history[-10:])) if state.loss_history else 0.0,
            'reward': float(np.mean(state.reward_history[-10:])) if state.reward_history else 0.0,
            'iterations': state.iteration_count
        }
        
        # Save model
        metadata = version_manager.save_model(
            model=state.oscillator,
            version=version,
            metrics=metrics,
            config={'field_size': state.field_size},
            training_info={'iterations': state.iteration_count},
            description=description,
            tags=tags
        )
        
        state.current_version = version
        
        return jsonify({
            'success': True,
            'version': version,
            'metadata': {
                'created_at': metadata.created_at,
                'metrics': metadata.metrics,
                'file_size': metadata.file_size
            }
        })
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/load/<version>', methods=['POST'])
@rate_limit
@require_permission('admin')
def load_model_version(version):
    """Load specific model version"""
    if not version_manager or not state.oscillator:
        return jsonify({'error': 'Model or versioning not available'}), 503
    
    try:
        metadata = version_manager.load_model(state.oscillator, version)
        state.current_version = version
        
        return jsonify({
            'success': True,
            'version': version,
            'metadata': {
                'created_at': metadata.created_at,
                'metrics': metadata.metrics
            }
        })
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/promote/<version>', methods=['POST'])
@rate_limit
@require_permission('admin')
def promote_model(version):
    """Promote model version to production"""
    if not version_manager:
        return jsonify({'error': 'Model versioning not available'}), 503
    
    try:
        version_manager.promote_to_production(version)
        return jsonify({
            'success': True,
            'production_version': version
        })
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitoring/stats', methods=['GET'])
@rate_limit
def get_monitoring_stats():
    """Get monitoring statistics"""
    if not model_monitor:
        return jsonify({'error': 'Monitoring not available'}), 503
    
    try:
        stats = model_monitor.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitoring/drift', methods=['GET'])
@rate_limit
def check_drift():
    """Check for model drift"""
    if not model_monitor:
        return jsonify({'error': 'Monitoring not available'}), 503
    
    try:
        drift_result = model_monitor.detect_drift()
        return jsonify(drift_result)
    except Exception as e:
        logger.error(f"Failed to check drift: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitoring/baseline', methods=['POST'])
@rate_limit
@require_permission('admin')
def set_monitoring_baseline():
    """Set current metrics as baseline"""
    if not model_monitor:
        return jsonify({'error': 'Monitoring not available'}), 503
    
    try:
        model_monitor.set_baseline()
        return jsonify({'success': True, 'message': 'Baseline set'})
    except Exception as e:
        logger.error(f"Failed to set baseline: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mode', methods=['GET'])
def get_model_mode():
    """Get current model mode"""
    if not UTILS_AVAILABLE:
        return jsonify({'mode': 'unknown'})
    
    return jsonify({'mode': mode_manager.mode.value})


@app.route('/api/mode/<mode>', methods=['POST'])
@rate_limit
@require_permission('admin')
def set_model_mode(mode):
    """Set model mode"""
    if not UTILS_AVAILABLE:
        return jsonify({'error': 'Mode management not available'}), 503
    
    try:
        mode_enum = ModelMode(mode)
        mode_manager.set_mode(mode_enum)
        return jsonify({
            'success': True,
            'mode': mode_enum.value
        })
    except ValueError:
        return jsonify({'error': 'Invalid mode'}), 400
    except Exception as e:
        logger.error(f"Failed to set mode: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health/detailed', methods=['GET'])
def detailed_health():
    """Detailed health check including model integrity"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': {
            'initialized': state.is_initialized,
            'running': state.is_running,
            'iterations': state.iteration_count
        }
    }
    
    # Model health
    if health_checker:
        try:
            model_health = health_checker.check_health()
            health_status['model'] = model_health
            if not model_health.get('healthy', False):
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['model'] = {'error': str(e)}
            health_status['status'] = 'degraded'
    
    # Current version
    if version_manager:
        try:
            prod_version = version_manager.get_production_version()
            health_status['version'] = {
                'current': state.current_version,
                'production': prod_version
            }
        except Exception as e:
            health_status['version'] = {'error': str(e)}
    
    # Mode
    if UTILS_AVAILABLE:
        health_status['mode'] = mode_manager.mode.value
    
    return jsonify(health_status)


@app.route('/api/auth/token', methods=['POST'])
@rate_limit
def generate_auth_token():
    """Generate authentication token"""
    if not UTILS_AVAILABLE:
        return jsonify({'error': 'Auth not available'}), 503
    
    from utils.security import auth_manager
    if not auth_manager or not auth_manager.enabled:
        return jsonify({'error': 'Auth not enabled'}), 503
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        role = data.get('role', 'user')
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        token = auth_manager.generate_token(user_id, role)
        return jsonify({'token': token})
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    host = getattr(config, 'HOST', '0.0.0.0')
    port = getattr(config, 'PORT', 5000)
    debug = getattr(config, 'DEBUG', False)
    
    # Initialize auth if available
    if UTILS_AVAILABLE:
        try:
            init_auth(app)
            logger.info("Authentication initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize auth: {e}")
    
    # Set initial mode
    if UTILS_AVAILABLE:
        mode_manager.enter_inference_mode()
        logger.info("Model mode set to inference")
    
    logger.info(f"Starting CognitionSim Benchmark Dashboard...")
    logger.info(f"Environment: {getattr(config, 'FLASK_ENV', 'development')}")
    logger.info(f"Open your browser to: http://{host}:{port}")
    
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
