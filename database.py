"""
Database models and persistence layer for CognitionSim
SQLAlchemy-based database for state persistence
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class SystemState(Base):
    """System state snapshots"""
    __tablename__ = 'system_states'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    field_size = Column(Integer, nullable=False)
    iteration_count = Column(Integer, default=0)
    is_running = Column(Boolean, default=False)
    is_initialized = Column(Boolean, default=False)
    current_version = Column(String(100), nullable=True)
    
    # Serialized data as JSON
    loss_history = Column(Text, nullable=True)  # JSON array
    reward_history = Column(Text, nullable=True)  # JSON array
    variance_history = Column(Text, nullable=True)  # JSON array
    field_mean_history = Column(Text, nullable=True)  # JSON array
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'field_size': self.field_size,
            'iteration_count': self.iteration_count,
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'current_version': self.current_version,
            'loss_history': json.loads(self.loss_history) if self.loss_history else [],
            'reward_history': json.loads(self.reward_history) if self.reward_history else [],
            'variance_history': json.loads(self.variance_history) if self.variance_history else [],
            'field_mean_history': json.loads(self.field_mean_history) if self.field_mean_history else [],
        }


class TrainingMetrics(Base):
    """Training metrics log"""
    __tablename__ = 'training_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    batch_number = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    reward = Column(Float, nullable=False)
    variance = Column(Float, nullable=False)
    field_mean = Column(Float, nullable=False)
    learning_rate = Column(Float, nullable=True)
    k_clusters = Column(Integer, nullable=True)
    speedup_factor = Column(Float, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'batch_number': self.batch_number,
            'loss': self.loss,
            'reward': self.reward,
            'variance': self.variance,
            'field_mean': self.field_mean,
            'learning_rate': self.learning_rate,
            'k_clusters': self.k_clusters,
            'speedup_factor': self.speedup_factor,
        }


class ModelVersion(Base):
    """Model version registry"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    version = Column(String(100), unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    sha256_hash = Column(String(64), nullable=True)
    file_path = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    is_production = Column(Boolean, default=False)
    metrics = Column(Text, nullable=True)  # JSON object
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'version': self.version,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'sha256_hash': self.sha256_hash,
            'file_path': self.file_path,
            'description': self.description,
            'is_production': self.is_production,
            'metrics': json.loads(self.metrics) if self.metrics else {},
        }


class Database:
    """Database manager for CognitionSim"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            database_url: SQLAlchemy database URL (default: sqlite:///atlantean_hrm.db)
        """
        if database_url is None:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///atlantean_hrm.db')
        
        logger.info(f"Initializing database: {database_url}")
        
        # Create engine
        self.engine = create_engine(database_url, echo=False)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
        # Create tables
        self.init_db()
    
    def init_db(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def save_system_state(self, state_dict: Dict[str, Any]) -> int:
        """
        Save system state snapshot
        
        Args:
            state_dict: Dictionary containing state data
            
        Returns:
            ID of saved state
        """
        with self.get_session() as session:
            state = SystemState(
                field_size=state_dict.get('field_size', 100),
                iteration_count=state_dict.get('iteration_count', 0),
                is_running=state_dict.get('is_running', False),
                is_initialized=state_dict.get('is_initialized', False),
                current_version=state_dict.get('current_version'),
                loss_history=json.dumps(state_dict.get('loss_history', [])),
                reward_history=json.dumps(state_dict.get('reward_history', [])),
                variance_history=json.dumps(state_dict.get('variance_history', [])),
                field_mean_history=json.dumps(state_dict.get('field_mean_history', [])),
            )
            session.add(state)
            session.flush()
            return state.id
    
    def get_latest_system_state(self) -> Optional[Dict[str, Any]]:
        """Get the most recent system state"""
        with self.get_session() as session:
            state = session.query(SystemState).order_by(SystemState.timestamp.desc()).first()
            return state.to_dict() if state else None
    
    def save_training_metric(self, metric_dict: Dict[str, Any]) -> int:
        """
        Save training metric
        
        Args:
            metric_dict: Dictionary containing metric data
            
        Returns:
            ID of saved metric
        """
        with self.get_session() as session:
            metric = TrainingMetrics(**metric_dict)
            session.add(metric)
            session.flush()
            return metric.id
    
    def get_training_metrics(self, limit: int = 100) -> list:
        """Get recent training metrics"""
        with self.get_session() as session:
            metrics = session.query(TrainingMetrics)\
                .order_by(TrainingMetrics.timestamp.desc())\
                .limit(limit)\
                .all()
            return [m.to_dict() for m in metrics]
    
    def save_model_version(self, version_dict: Dict[str, Any]) -> int:
        """
        Save model version
        
        Args:
            version_dict: Dictionary containing version data
            
        Returns:
            ID of saved version
        """
        with self.get_session() as session:
            version = ModelVersion(
                version=version_dict['version'],
                sha256_hash=version_dict.get('sha256_hash'),
                file_path=version_dict['file_path'],
                description=version_dict.get('description'),
                is_production=version_dict.get('is_production', False),
                metrics=json.dumps(version_dict.get('metrics', {})),
            )
            session.add(version)
            session.flush()
            return version.id
    
    def get_model_versions(self) -> list:
        """Get all model versions"""
        with self.get_session() as session:
            versions = session.query(ModelVersion)\
                .order_by(ModelVersion.timestamp.desc())\
                .all()
            return [v.to_dict() for v in versions]
    
    def promote_model_version(self, version: str) -> bool:
        """
        Promote a model version to production
        
        Args:
            version: Version string to promote
            
        Returns:
            True if successful
        """
        with self.get_session() as session:
            # Demote all existing production versions
            session.query(ModelVersion).update({'is_production': False})
            
            # Promote the specified version
            model = session.query(ModelVersion).filter_by(version=version).first()
            if model:
                model.is_production = True
                return True
            return False


# Global database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """Get or create global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def init_database(database_url: Optional[str] = None):
    """Initialize global database instance"""
    global _db_instance
    _db_instance = Database(database_url)
    return _db_instance
