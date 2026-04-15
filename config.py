"""
Environment-based configuration for CognitionSim
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Base configuration class"""
    
    # Application
    APP_NAME = "CognitionSim"
    VERSION = "1.0.0"
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Paths
    BASE_DIR = Path(__file__).parent.resolve()
    MODEL_ARTIFACTS_DIR = Path(os.getenv('MODEL_ARTIFACTS_DIR', BASE_DIR / 'models'))
    DASHBOARD_STATE_DIR = Path(os.getenv('DASHBOARD_STATE_DIR', BASE_DIR / 'dashboard_state'))
    LOGS_DIR = Path(os.getenv('LOGS_DIR', BASE_DIR / 'logs'))
    DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
    
    # Create directories
    MODEL_ARTIFACTS_DIR.mkdir(exist_ok=True)
    DASHBOARD_STATE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Model Configuration
    FIELD_SIZE = int(os.getenv('FIELD_SIZE', 100))
    NUM_BATCHES = int(os.getenv('NUM_BATCHES', 100))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 5))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    
    # Device
    DEVICE = os.getenv('DEVICE', 'cpu')
    
    # Dataset
    DATASET_NAME = os.getenv('DATASET_NAME', 'wikitext')
    DATASET_CONFIG = os.getenv('DATASET_CONFIG', 'wikitext-2-raw-v1')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Model Artifacts
    OSCILLATOR_WEIGHTS_FILE = 'oscillator_weights.pth'
    ATLANTEAN_HRM_ENHANCED_FILE = 'atlantean_hrm_enhanced.pth'
    ATLANTEAN_HRM_WEIGHTS_FILE = 'atlantean_hrm_weights.pth'
    TRAINING_METRICS_FILE = 'training_metrics.json'
    
    # Socket.IO
    SOCKETIO_MESSAGE_QUEUE = os.getenv('SOCKETIO_MESSAGE_QUEUE', None)
    SOCKETIO_ASYNC_MODE = os.getenv('SOCKETIO_ASYNC_MODE', 'eventlet')
    
    # Performance
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
    SAVE_INTERVAL = int(os.getenv('SAVE_INTERVAL', 10))
    
    @classmethod
    def get_model_path(cls, filename: str) -> Path:
        """Get full path for model artifact"""
        return cls.MODEL_ARTIFACTS_DIR / filename
    
    @classmethod
    def get_state_path(cls, filename: str) -> Path:
        """Get full path for dashboard state"""
        return cls.DASHBOARD_STATE_DIR / filename
    
    @classmethod
    def get_log_path(cls, filename: str) -> Path:
        """Get full path for log file"""
        return cls.LOGS_DIR / filename


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    LOG_LEVEL = 'WARNING'
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'https://yourdomain.com')  # Restrict in production
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///atlantean_hrm.db')
    
    # Require secret key in production
    @classmethod
    def validate(cls):
        secret_key = os.getenv('SECRET_KEY', cls.SECRET_KEY)
        if secret_key == 'dev-secret-key-change-in-production':
            raise ValueError("Must set SECRET_KEY in production environment")
        if not os.getenv('SECRET_KEY'):
            raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    NUM_BATCHES = 5
    BATCH_SIZE = 2


# Configuration map
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    config_class = config_map.get(env, DevelopmentConfig)
    
    # Validate production config
    if env == 'production':
        config_class.validate()
    
    return config_class
