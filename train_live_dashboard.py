#!/usr/bin/env python3
"""
Live Training Dashboard - WikiText Training with Real-time Visualization
Runs training in background and streams metrics to web dashboard via WebSocket
"""

import asyncio
import logging
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for metrics
training_state = {
    'active': False,
    'batch': 0,
    'total_batches': 50,
    'metrics': {
        'speedup_factors': [],
        'rewards': [],
        'learning_rates': [],
        'k_values': [],
        'cache_efficiency': [],
        'field_resonance': [],
        'timestamps': []
    },
    'status': 'initializing',
    'start_time': None,
    'current_batch_progress': 0,
    'eta_seconds': 0
}

def get_training_metrics():
    """Return current training metrics for dashboard"""
    elapsed = 0
    if training_state['start_time']:
        elapsed = time.time() - training_state['start_time']
    
    metrics = training_state['metrics']
    
    return {
        'active': training_state['active'],
        'status': training_state['status'],
        'batch': training_state['batch'],
        'total_batches': training_state['total_batches'],
        'progress_percent': (training_state['batch'] / training_state['total_batches']) * 100,
        'elapsed_seconds': elapsed,
        'eta_seconds': training_state['eta_seconds'],
        'speedup_current': metrics['speedup_factors'][-1] if metrics['speedup_factors'] else 0,
        'speedup_peak': max(metrics['speedup_factors']) if metrics['speedup_factors'] else 0,
        'speedup_avg': np.mean(metrics['speedup_factors']) if metrics['speedup_factors'] else 0,
        'reward_current': metrics['rewards'][-1] if metrics['rewards'] else 0,
        'reward_trend': metrics['rewards'][-5:] if len(metrics['rewards']) >= 5 else metrics['rewards'],
        'learning_rate': metrics['learning_rates'][-1] if metrics['learning_rates'] else 0.001,
        'k_clusters': int(metrics['k_values'][-1]) if metrics['k_values'] else 3,
        'cache_efficiency': metrics['cache_efficiency'][-1] if metrics['cache_efficiency'] else 0,
        'field_resonance': metrics['field_resonance'][-1] if metrics['field_resonance'] else 0,
        'all_speedup_factors': metrics['speedup_factors'],
        'all_rewards': metrics['rewards'],
        'all_learning_rates': metrics['learning_rates'],
        'all_k_values': metrics['k_values'],
        'all_cache_efficiency': metrics['cache_efficiency'],
        'all_field_resonance': metrics['field_resonance'],
    }

def update_metrics(batch_id: int, metrics: Dict):
    """Update global training state with new batch metrics"""
    training_state['batch'] = batch_id
    training_state['metrics']['speedup_factors'].append(float(metrics['speedup_factors'][-1]))
    training_state['metrics']['rewards'].append(float(metrics['rewards'][-1]))
    training_state['metrics']['learning_rates'].append(float(metrics['learning_rates'][-1]))
    training_state['metrics']['k_values'].append(float(metrics['k_values'][-1]))
    training_state['metrics']['cache_efficiency'].append(float(metrics['cache_efficiency'][-1]))
    training_state['metrics']['field_resonance'].append(float(metrics['field_resonance'][-1]))
    training_state['metrics']['timestamps'].append(datetime.now().isoformat())
    
    # Calculate ETA
    if batch_id > 0 and training_state['start_time']:
        elapsed = time.time() - training_state['start_time']
        rate = elapsed / batch_id
        remaining_batches = training_state['total_batches'] - batch_id
        training_state['eta_seconds'] = rate * remaining_batches


async def run_training():
    """Run WikiText training with live metric updates"""
    
    training_state['active'] = True
    training_state['status'] = 'loading_dataset'
    training_state['start_time'] = time.time()
    
    try:
        logger.info("Initializing CognitionSimTrainer...")
        from train_atlantean_hrm import CognitionSimTrainer
        
        training_state['status'] = 'training'
        
        trainer = CognitionSimTrainer(
            field_size=150,
            device='cpu',
            enable_noise=True,
            noise_intensity=0.15
        )
        
        logger.info("Starting training on WikiText-2...")
        
        # Train and sync metrics continuously
        import sys
        from datasets import load_dataset
        
        # Load dataset
        logger.info("Loading WikiText-2 dataset...")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=True)
            logger.info("✅ Dataset loaded in streaming mode")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            dataset = None
        
        if dataset:
            training_state['status'] = 'training'
            dataset_iter = iter(dataset)
            
            # Training loop with continuous metric updates
            for batch_id in range(training_state['total_batches']):
                texts = []
                
                # Collect batch from stream
                for _ in range(10):  # batch_size = 10
                    try:
                        sample = next(dataset_iter)
                        if 'text' in sample:
                            text = sample['text']
                        elif 'content' in sample:
                            text = sample['content']
                        else:
                            text = str(sample)[:500]
                        
                        if text and len(text.strip()) > 10:
                            texts.append(text)
                    except StopIteration:
                        dataset_iter = iter(dataset)
                        break
                
                if not texts:
                    continue
                
                # Train on batch
                logger.info(f"Training batch {batch_id + 1}/{training_state['total_batches']}...")
                batch_metrics = await trainer.train_on_text_batch(texts, batch_id)
                
                # Sync trainer metrics to global state
                if hasattr(trainer, 'metrics'):
                    trainer_metrics = trainer.metrics
                    training_state['metrics']['speedup_factors'] = [float(x) for x in trainer_metrics.get('speedup_factors', [])]
                    training_state['metrics']['rewards'] = [float(x) for x in trainer_metrics.get('rewards', [])]
                    training_state['metrics']['learning_rates'] = [float(x) for x in trainer_metrics.get('learning_rates', [])]
                    training_state['metrics']['k_values'] = [float(x) for x in trainer_metrics.get('k_values', [])]
                    training_state['metrics']['cache_efficiency'] = [float(x) for x in trainer_metrics.get('cache_efficiency', [])]
                    training_state['metrics']['field_resonance'] = [float(x) for x in trainer_metrics.get('field_resonance', [])]
                    
                    training_state['batch'] = batch_id + 1
                    
                    # Calculate ETA
                    elapsed = time.time() - training_state['start_time']
                    if batch_id > 0:
                        rate = elapsed / (batch_id + 1)
                        remaining_batches = training_state['total_batches'] - (batch_id + 1)
                        training_state['eta_seconds'] = rate * remaining_batches
                    
                    latest_reward = training_state['metrics']['rewards'][-1] if training_state['metrics']['rewards'] else 0
                    logger.info(f"✅ Batch {batch_id + 1} synced: {len(training_state['metrics']['rewards'])} samples, reward={latest_reward:.3f}")
                else:
                    logger.warning(f"Trainer has no metrics attribute at batch {batch_id + 1}")
        
        training_state['status'] = 'saving'
        
        # Save model
        save_path = "atlantean_hrm_wikitext.pth"
        trainer.oscillator.save_weights(save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Save metrics
        metrics_file = "training_metrics_wikitext.json"
        with open(metrics_file, 'w') as f:
            json.dump(training_state['metrics'], f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
        
        training_state['status'] = 'complete'
        logger.info("✅ Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        training_state['status'] = f'error: {str(e)}'
    finally:
        training_state['active'] = False


def start_training_background():
    """Start training in background thread"""
    def run_async():
        asyncio.run(run_training())
    
    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()
    return thread


def create_live_dashboard_app():
    """Create Flask app with WebSocket support for live metrics"""
    from flask import Flask, render_template, jsonify
    from flask_socketio import SocketIO, emit, disconnect
    
    app = Flask(__name__, template_folder='templates')
    app.config['SECRET_KEY'] = 'cognitionsim-training-live'
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*",
        async_mode='threading',
        ping_timeout=60,
        ping_interval=25
    )
    
    training_thread = None
    
    @app.route('/')
    def index():
        return render_template('wikitext_dashboard.html')
    
    @app.route('/api/training/status')
    def get_status():
        return jsonify(get_training_metrics())
    
    @socketio.on('connect')
    def handle_connect():
        logger.info('Client connected to dashboard')
        # Send initial status immediately with all current metrics
        current_metrics = get_training_metrics()
        logger.debug(f"Sending initial metrics to new client: {len(current_metrics.get('all_rewards', []))} reward data points")
        emit('status_update', current_metrics)
        emit('message', {'data': 'Connected to training server'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info('Client disconnected from dashboard')
    
    @socketio.on('start_training')
    def handle_start_training():
        nonlocal training_thread
        if not training_state['active']:
            logger.info('Starting training from dashboard...')
            training_thread = start_training_background()
            emit('training_started', {'message': 'Training started'})
    
    @socketio.on('stop_training')
    def handle_stop_training():
        logger.info('Stop training requested (graceful shutdown not implemented)')
        emit('message', {'data': 'Stop requested'})
    
    # Background thread for pushing updates to all connected clients
    def background_push_updates():
        logger.info("Background update thread started")
        while True:
            try:
                metrics = get_training_metrics()
                socketio.emit('status_update', metrics)
                time.sleep(0.5)  # Update every 500ms for smooth animation
            except Exception as e:
                logger.debug(f"Background emit error: {e}")
                time.sleep(1)
    
    # Start background thread immediately
    bg_thread = threading.Thread(target=background_push_updates, daemon=True)
    bg_thread.start()
    logger.info("Background update thread initialized")
    
    return app, socketio, start_training_background


async def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🚀 QUADRA MATRIX - LIVE TRAINING DASHBOARD")
    print("="*80)
    print("\n📊 Configuration:")
    print("  • Dataset: WikiText-2 (raw)")
    print("  • Field size: 150")
    print("  • Training batches: 50")
    print("  • Batch size: 10 samples")
    print("  • GUI: Flask + WebSocket live dashboard")
    print("\n🌐 Dashboard Available at: http://localhost:5000")
    print("="*80 + "\n")
    
    try:
        # Create Flask app with WebSocket support
        app, socketio, start_training = create_live_dashboard_app()
        
        # Start training in background
        print("📈 Starting training in background...")
        start_training()
        
        # Start Flask server
        print("🌐 Starting dashboard server on http://localhost:5000\n")
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
