# Neural Command Center - Architecture Documentation

## System Overview

The Neural Command Center is a production-grade web dashboard implementing the **Three-Plane Cognitive Architecture** for real-time AI cognition monitoring and interaction.

---

## Three-Plane Architecture

### Design Philosophy

**Never hide cognition behind tabs or modals** - all three planes are always visible, creating a holistic view of the cognitive process.

### Plane 1: Cognitive Physiology (Top-Left)

**Purpose**: Visualize the internal cognitive field dynamics

**Components**:
- **4 Cognitive Layers** (quadrants):
  1. Perception (top-left) - Sensory input processing
  2. Integration (top-right) - Cross-modal synthesis
  3. Reasoning (bottom-left) - Inference and deliberation
  4. Action (bottom-right) - Motor/output generation

**Visual Elements**:
- **Pulses**: Intensity of neural activity (circle size)
- **Flows**: Information transfer (particle streams)
- **Heat Maps**: Activity concentration (blue → yellow → red)
- **Conflicts**: Competing representations (red highlights)

**Rendering**: HTML5 Canvas with 60fps real-time updates

### Plane 2: Intent & Deliberation (Top-Right)

**Purpose**: Show goal-directed reasoning and hypothesis testing

**Components**:
- **Goal Nodes**: Circles with priority-based size
- **Hypotheses**: Orbiting satellites around goals
- **Confidence Mass**: Visual indicator of certainty distribution

**Interactions**:
- **Click to pin**: Prioritize/freeze a goal node
- **Hover**: View hypothesis details
- **Drag** (future): Manually adjust goal positions

**Rendering**: Canvas with physics-based orbital dynamics

### Plane 3: World Interface (Bottom)

**Purpose**: Bridge between cognitive system and external environment

**Components**:
- **Sensors** (left):
  - Visual stream (RGB, depth, optical flow)
  - Temporal stream (event sequences)
  - Pattern stream (latent features)
  
- **Actions** (right):
  - Proposed action vectors with uncertainty
  - Human approval buttons (✓ / ✗)
  - Action history log

**Interactions**:
- **Approve/Reject**: Human-in-the-loop control
- **Sensor injection** (future): Manual override

**Rendering**: React components with Tailwind CSS

### Interaction Panel (Bottom-Right)

**Purpose**: Direct cognitive intervention without chat

**Tools**:
- **Signal Injection**: Send raw sensory data
- **Cognitive Probes**: Query uncertainty, conflicts, attention
- **Safety Governors**: Apply constraints/policies

**Why No Chat?**
Chat implies language; cognition is multi-modal. Instead, we use:
- Vector signals
- Policy constraints
- Direct field manipulation

---

## Technical Architecture

### Backend Stack

```
FastAPI (0.104.1)
├── Uvicorn ASGI Server
├── WebSockets (real-time streaming)
├── Pydantic (data validation)
└── CognitionDemo (cognitive engine)
```

**Key Design Decisions**:
- **Async/await**: All endpoints are async for non-blocking I/O
- **Background tasks**: Cognition loop runs independently
- **State management**: In-memory DashboardState class (Redis future)
- **WebSocket broadcast**: All connected clients receive updates

### Frontend Stack

```
React 18.2.0
├── Vite 5.0.4 (build tool)
├── Zustand 4.4.7 (state management)
├── Tailwind CSS 3.3.6 (styling)
├── Recharts 2.10.3 (training charts)
└── Axios 1.6.2 (HTTP client)
```

**Key Design Decisions**:
- **Zustand over Redux**: Simpler API, less boilerplate
- **Canvas over SVG**: Better performance for real-time rendering
- **Vite over CRA**: Faster HMR, smaller bundles
- **Functional components**: All hooks, no classes

### Data Flow

```
Cognitive Engine
    ↓ (asyncio task)
WebSocket Server
    ↓ (broadcast)
Frontend WebSocket Client
    ↓ (Zustand store update)
React Components
    ↓ (Canvas/DOM rendering)
User Display
```

---

## API Design

### REST Principles

- **Stateless**: Each request contains all needed info
- **Resource-oriented**: `/api/{resource}/{id}`
- **HTTP methods**: GET, POST, PUT, DELETE
- **JSON responses**: Consistent structure

### WebSocket Protocol

**Connection**: `ws://localhost:8000/ws/cognition`

**Message Format**:
```json
{
  "type": "cognition_update",
  "data": {
    "field": [[...]],  // 100x100 field
    "attention": [0.5, 0.5],
    "uncertainty": 0.3,
    "conflicts": [...],
    "goals": [...],
    "sensors": {...},
    "proposed_actions": [...]
  },
  "timestamp": 1234567890
}
```

**Update Rate**: 10 Hz (100ms intervals)

---

## Deployment Architecture

### Development

```
[Frontend Vite Dev Server :5173]
           ↓ proxy
[Backend FastAPI :8000]
```

### Docker Compose

```
[Nginx :3000] ← Frontend static files
    ↓ /api/*  ← Reverse proxy
    ↓ /ws/*   ← WebSocket upgrade
[Backend :8000] ← Internal network
```

### Kubernetes (Future)

```
[Ingress Controller]
    ↓
[Frontend Service]
    ↓
[Frontend Pods × N]

[Backend Service]
    ↓
[Backend Pods × N]
    ↓
[Redis Cluster]
[PostgreSQL]
```

---

## Security Considerations

### Current State (MVP)

- ✅ CORS configured
- ✅ Input validation (Pydantic)
- ⚠️  No authentication
- ⚠️  No rate limiting
- ⚠️  HTTP only (no HTTPS)

### Production Requirements

- [ ] JWT/OAuth2 authentication
- [ ] Role-based access control (RBAC)
- [ ] API rate limiting
- [ ] Request signing
- [ ] HTTPS/WSS encryption
- [ ] Audit logging
- [ ] Secret management (Vault)

---

## Performance Optimization

### Backend

- **Async I/O**: Non-blocking WebSocket broadcasts
- **Efficient serialization**: Torch tensors → NumPy → JSON
- **Background tasks**: Cognition loop doesn't block API
- **Connection pooling** (future): Database connections

### Frontend

- **Canvas rendering**: Direct pixel manipulation (faster than DOM)
- **RequestAnimationFrame**: Smooth 60fps updates
- **Zustand**: Minimal re-renders, subscription-based
- **Code splitting**: Lazy load views
- **Asset optimization**: Vite tree-shaking, minification

### Network

- **WebSocket compression**: gzip extension
- **Binary protocols** (future): MessagePack or Protobuf
- **CDN** (future): Static asset delivery
- **HTTP/2**: Multiplexing, server push

---

## Scalability

### Horizontal Scaling

**Backend**:
- **Stateless API**: Can run multiple instances
- **WebSocket sticky sessions**: Nginx `ip_hash` or Redis Pub/Sub
- **Shared state**: Redis for cross-pod communication

**Frontend**:
- **Static files**: Infinite scaling via CDN
- **API gateway**: Load balancer distributes requests

### Vertical Scaling

- **GPU acceleration**: Move cognition engine to GPU workers
- **Larger fields**: Sparse tensors for 1000×1000+ fields
- **Batch processing**: Group WebSocket messages

---

## Monitoring & Observability

### Metrics (Future)

- **Prometheus** export:
  - Request latency histogram
  - WebSocket connection count
  - Cognition loop FPS
  - Memory usage
  
- **Grafana** dashboards:
  - System health
  - User activity
  - Cognitive performance

### Logging

- **Structured JSON logs**
- **Correlation IDs**: Track requests across services
- **Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Tracing

- **OpenTelemetry**: Distributed tracing
- **Jaeger/Zipkin**: Trace visualization

---

## Future Enhancements

### Multi-User Features

- **User authentication**: Sign up, login, sessions
- **Workspaces**: Isolated cognitive environments per user/team
- **Sharing**: Invite collaborators, permissions
- **Session recording**: Save and replay cognition sessions

### Advanced Visualizations

- **3D rendering**: Three.js or Babylon.js for volumetric fields
- **VR/AR**: Immersive cognitive exploration
- **Time travel**: Scrub through historical states
- **Comparison mode**: Side-by-side A/B testing

### Edge Computing

- **Edge SDK**: Python/C++ client for IoT devices
- **Model compression**: Quantization, pruning for edge
- **Federated learning**: Aggregate updates from edge nodes
- **Offline mode**: Local cognition when disconnected

### Cognitive Enhancements

- **Memory persistence**: Long-term episodic storage
- **Transfer learning**: Pre-trained cognitive modules
- **Multi-modal fusion**: Vision + audio + text + proprioception
- **Meta-cognition**: AI reflecting on its own thinking

---

## Integration Points

### Existing Quadra Components

| Component | Integration Method | Status |
|-----------|-------------------|--------|
| `train_live_dashboard.py` | POST to `/api/training/metrics` | ✅ Ready |
| `demo_cognition.py` | Imported as cognitive engine | ✅ Active |
| `dashboard.html` | Deprecated (replaced by React) | 🔄 Migrated |
| `model_monitoring.py` | POST to `/api/edge/devices` | ✅ Ready |
| `tails_memory.py` | Future persistence layer | ⏳ Planned |

### External Systems

- **MLflow**: Experiment tracking integration
- **DVC**: Dataset version control
- **Weights & Biases**: Training visualization
- **TensorBoard**: Legacy support
- **Jupyter**: Notebook-based exploration

---

## Development Workflow

### Local Development

1. **Backend**: `cd dashboard/backend && python main.py`
2. **Frontend**: `cd dashboard/frontend && npm run dev`
3. **Access**: http://localhost:5173

### Docker Development

```bash
docker-compose up
# Edit code → Auto-reload
# View logs: docker-compose logs -f
```

### Production Build

```bash
cd dashboard/frontend
npm run build
# Outputs to dist/
# Serve with nginx, Apache, or CDN
```

---

## Testing Strategy

### Unit Tests

- **Backend**: `pytest` for API endpoints, cognitive logic
- **Frontend**: `vitest` for component logic, store mutations

### Integration Tests

- **API tests**: `pytest` + `httpx` for full request/response cycles
- **WebSocket tests**: Test client connection, message handling

### E2E Tests

- **Playwright** or **Cypress**: Full user flows
- **Visual regression**: Screenshot comparison

### Performance Tests

- **Locust**: Load testing for concurrent users
- **Artillery**: WebSocket stress testing

---

## Troubleshooting

### WebSocket Connection Failed

**Symptoms**: "Disconnected" badge, no real-time updates

**Solutions**:
1. Check backend is running: `curl http://localhost:8000/`
2. Verify WebSocket upgrade headers in nginx config
3. Check browser console for errors
4. Try incognito mode (disable extensions)

### Canvas Not Rendering

**Symptoms**: Blank white squares in Cognition view

**Solutions**:
1. Check browser console for errors
2. Verify `data.field` exists in WebSocket message
3. Test with smaller field size (10×10)
4. Ensure Canvas API is supported (not IE11)

### High CPU Usage

**Symptoms**: Fan spinning, browser lag

**Solutions**:
1. Reduce update rate: `VITE_WS_RECONNECT_DELAY=5000`
2. Lower Canvas FPS: `VITE_CANVAS_FPS=30`
3. Limit WebSocket connections (one tab only)
4. Use production build (optimized)

---

**Last Updated**: 2025-01-XX  
**Maintainer**: CognitionSim Team
