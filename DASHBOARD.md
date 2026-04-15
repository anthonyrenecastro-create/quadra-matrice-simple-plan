# CognitionSim A.I. Dashboard

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_dashboard.txt
```

### 2. Run the Dashboard
```bash
python app.py
```

### 3. Open in Browser
Navigate to: **http://localhost:5000**

## Features

### Real-Time Monitoring
- **Live Metrics Display**: Loss, Reward, Variance, Field Mean, Q-Table Size
- **Interactive Charts**: Real-time updating line charts for all key metrics
- **System Status**: Visual indicators showing system state (Not Initialized, Ready, Training)
- **Live Logs**: Stream of training events and system messages

### Control Panel
- **Initialize System**: Set up all CognitionSim components (Oscillator, Core Field, Syntropy Engine, etc.)
- **Start Training**: Begin the training loop with streaming text data
- **Stop Training**: Pause training while maintaining state
- **Reset System**: Clear all data and return to initial state

### Architecture
- **Backend**: Flask + Socket.IO for real-time bidirectional communication
- **Frontend**: Vanilla JavaScript with Chart.js for live visualizations
- **Threading**: Background training loop doesn't block the web server
- **WebSocket**: Efficient real-time updates pushed to all connected clients

## Dashboard Components

### Status Cards
- System Status (with visual indicator)
- Iteration Count
- Current Loss
- Current Reward
- Field Variance
- Q-Table Size

### Charts (100 data points rolling window)
- Loss History (red gradient)
- Reward History (green gradient)
- Field Variance (purple gradient)
- Field Mean (cyan gradient)

### System Logs
- Scrollable log viewer
- Color-coded by message type (info, success, warning, error)
- Timestamps for all events
- Auto-limited to last 50 entries

## Training Process

The dashboard runs the CognitionSim training loop with:
- 8 diverse test texts covering quantum, neural, and AI concepts
- 10 synthetic samples per iteration
- 3 training epochs per batch
- Field updates with vibrational modes
- Syntropy regulation
- Q-learning with dynamic table

## Technical Details

### Socket.IO Events

**Client → Server:**
- `initialize_system`: Initialize all CognitionSim components
- `start_training`: Start the training loop
- `stop_training`: Stop the training loop
- `reset_system`: Reset all state

**Server → Client:**
- `status_message`: System notifications (info, success, warning, error)
- `system_initialized`: Initialization complete
- `training_status`: Training started/stopped
- `system_reset`: Reset complete
- `metrics_update`: Real-time metrics (every iteration)

### API Endpoints
- `GET /`: Dashboard HTML page
- `GET /api/status`: JSON status including current metrics and history

## Browser Compatibility
- Chrome/Edge (recommended)
- Firefox
- Safari
- Any modern browser with WebSocket support

## Performance
- Efficient data streaming with Socket.IO
- Chart animations optimized for 60 FPS
- Memory-efficient rolling data windows
- Non-blocking background training thread

## Customization

### Modify Training Texts
Edit the `test_texts` list in `app.py` training_loop()

### Adjust Chart History
Change `maxDataPoints` in `dashboard.html` (default: 100)

### Change Port
Modify the last line in `app.py`: `socketio.run(app, host='0.0.0.0', port=5000)`

### Field Size
Adjust `field_size` in `SystemState.__init__()` (default: 100)

## Troubleshooting

### Connection Issues
- Ensure port 5000 is not blocked by firewall
- Check if another service is using port 5000
- Try accessing via `http://127.0.0.1:5000`

### Training Errors
- Verify `atlantean_hrm_spi.py` is in the same directory
- Check all dependencies are installed
- Review system logs in the dashboard

### Performance Issues
- Reduce `maxDataPoints` in charts
- Increase `time.sleep()` delay in training loop
- Close other browser tabs

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│         Browser (Dashboard UI)              │
│  ┌──────────────────────────────────────┐   │
│  │  Controls | Status | Charts | Logs   │   │
│  └──────────────────────────────────────┘   │
└─────────────────┬───────────────────────────┘
                  │ WebSocket (Socket.IO)
┌─────────────────▼───────────────────────────┐
│         Flask + Socket.IO Server            │
│  ┌──────────────────────────────────────┐   │
│  │  Event Handlers | API Routes         │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────┐   │
│  │  Training Thread (Background)        │   │
│  └──────────────────────────────────────┘   │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         CognitionSim Components             │
│  • OscillatorySynapseTheory                 │
│  • CoreField                                │
│  • SyntropyEngine                           │
│  • PatternModule                            │
│  • NeuroplasticityManager                   │
│  • SymbolicPredictiveInterpreter            │
└─────────────────────────────────────────────┘
```

## License
Same as CognitionSim A.I. project
