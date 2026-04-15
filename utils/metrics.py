"""
Prometheus Metrics for CognitionSim
Custom metrics collection and exporting
"""
import time
from functools import wraps
from typing import Callable, Any
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import logging

logger = logging.getLogger(__name__)

# ============================================================
# APPLICATION METRICS
# ============================================================

# Request metrics
http_requests_total = Counter(
    'atlantean_hrm_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'atlantean_hrm_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# WebSocket metrics
websocket_connections = Gauge(
    'atlantean_hrm_websocket_connections',
    'Current WebSocket connections'
)

websocket_messages_total = Counter(
    'atlantean_hrm_websocket_messages_total',
    'Total WebSocket messages',
    ['event', 'direction']
)

# ============================================================
# ML/TRAINING METRICS
# ============================================================

# Training metrics
training_batches_total = Counter(
    'atlantean_hrm_training_batches_total',
    'Total training batches processed'
)

training_loss = Gauge(
    'atlantean_hrm_training_loss',
    'Current training loss'
)

training_reward = Gauge(
    'atlantean_hrm_training_reward',
    'Current training reward'
)

training_field_variance = Gauge(
    'atlantean_hrm_training_field_variance',
    'Current field variance'
)

training_duration_seconds = Histogram(
    'atlantean_hrm_training_duration_seconds',
    'Training batch duration'
)

# Model metrics
model_inference_duration_seconds = Histogram(
    'atlantean_hrm_model_inference_duration_seconds',
    'Model inference latency',
    ['model_version']
)

model_predictions_total = Counter(
    'atlantean_hrm_model_predictions_total',
    'Total model predictions',
    ['model_version']
)

active_training_sessions = Gauge(
    'atlantean_hrm_active_training_sessions',
    'Number of active training sessions'
)

# ============================================================
# SYSTEM METRICS
# ============================================================

# System state
system_state = Gauge(
    'atlantean_hrm_system_state',
    'System state (1=initialized, 2=training, 0=stopped)'
)

field_size = Gauge(
    'atlantean_hrm_field_size',
    'Current field size'
)

iteration_count = Gauge(
    'atlantean_hrm_iteration_count',
    'Total iterations processed'
)

# Database metrics
database_operations_total = Counter(
    'atlantean_hrm_database_operations_total',
    'Total database operations',
    ['operation', 'table']
)

database_operation_duration_seconds = Histogram(
    'atlantean_hrm_database_operation_duration_seconds',
    'Database operation latency',
    ['operation']
)

# Error metrics
errors_total = Counter(
    'atlantean_hrm_errors_total',
    'Total errors',
    ['error_type', 'component']
)

# Application info
app_info = Info(
    'atlantean_hrm_app',
    'Application information'
)

# ============================================================
# DECORATORS
# ============================================================

def track_request_metrics(func: Callable) -> Callable:
    """
    Decorator to track HTTP request metrics
    
    Usage:
        @app.route('/api/endpoint')
        @track_request_metrics
        def endpoint():
            return jsonify({'status': 'ok'})
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        status_code = 200
        
        try:
            result = func(*args, **kwargs)
            
            # Extract status code if response object
            if hasattr(result, 'status_code'):
                status_code = result.status_code
            elif isinstance(result, tuple) and len(result) > 1:
                status_code = result[1]
            
            return result
        except Exception as e:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            
            # Get endpoint name
            endpoint = func.__name__
            method = 'GET'  # Default, should be extracted from request
            
            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
    
    return wrapper


def track_training_metrics(func: Callable) -> Callable:
    """
    Decorator to track training metrics
    
    Usage:
        @track_training_metrics
        def training_step():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            training_batches_total.inc()
            return result
        finally:
            duration = time.time() - start_time
            training_duration_seconds.observe(duration)
    
    return wrapper


def track_inference_metrics(model_version: str = 'default'):
    """
    Decorator factory to track inference metrics
    
    Usage:
        @track_inference_metrics(model_version='v1.0.0')
        def predict(input_data):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                model_predictions_total.labels(model_version=model_version).inc()
                return result
            finally:
                duration = time.time() - start_time
                model_inference_duration_seconds.labels(
                    model_version=model_version
                ).observe(duration)
        
        return wrapper
    return decorator


def track_database_operation(operation: str, table: str = 'unknown'):
    """
    Decorator factory to track database operations
    
    Usage:
        @track_database_operation('insert', 'system_states')
        def save_state():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                database_operations_total.labels(
                    operation=operation,
                    table=table
                ).inc()
                return result
            except Exception as e:
                errors_total.labels(
                    error_type=type(e).__name__,
                    component='database'
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                database_operation_duration_seconds.labels(
                    operation=operation
                ).observe(duration)
        
        return wrapper
    return decorator


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def update_training_metrics(loss: float, reward: float, variance: float):
    """Update current training metrics"""
    training_loss.set(loss)
    training_reward.set(reward)
    training_field_variance.set(variance)


def update_system_metrics(state: int, field_sz: int, iterations: int):
    """Update system state metrics"""
    system_state.set(state)
    field_size.set(field_sz)
    iteration_count.set(iterations)


def record_error(error_type: str, component: str = 'unknown'):
    """Record an error occurrence"""
    errors_total.labels(error_type=error_type, component=component).inc()


def set_app_info(version: str, environment: str):
    """Set application metadata"""
    app_info.info({
        'version': version,
        'environment': environment,
        'name': 'CognitionSim'
    })


# ============================================================
# METRICS MANAGER
# ============================================================

class MetricsManager:
    """Central metrics management"""
    
    def __init__(self):
        self.initialized = False
    
    def initialize(self, app_version: str = '1.0.0', environment: str = 'production'):
        """Initialize metrics with app info"""
        if not self.initialized:
            set_app_info(app_version, environment)
            logger.info(f"Metrics initialized: {app_version} ({environment})")
            self.initialized = True
    
    def track_websocket_connection(self, connected: bool):
        """Track WebSocket connection changes"""
        if connected:
            websocket_connections.inc()
        else:
            websocket_connections.dec()
    
    def track_websocket_message(self, event: str, direction: str = 'incoming'):
        """Track WebSocket message"""
        websocket_messages_total.labels(event=event, direction=direction).inc()
    
    def track_training_session(self, active: bool):
        """Track training session state"""
        if active:
            active_training_sessions.inc()
        else:
            active_training_sessions.dec()


# Global metrics manager instance
metrics_manager = MetricsManager()
