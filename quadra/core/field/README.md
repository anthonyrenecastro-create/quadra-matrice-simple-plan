# IFT Field Engine - Ten-Field Substrate

**Interplayed Field Theory integration for CognitionSim**

---

## Overview

Implements a coupled system of 10 differential equations:

```
∂φₙ/∂t = Fₙ(φₙ, ∇²φₙ, ∇⁴φₙ) + ε·T_nmp(ω,x)·Σφₘ
```

where:
- **10 fields** φₙ (n=0..9) on spatial grid or scalar
- **Field evolution** Fₙ includes damping, saturation, stiffness
- **Resonant coupling** T_nmp with 12-fold angular symmetry
- **Adaptive control** via field-modulated thresholds and leak rates

**Key Features:**
- 12-fold angular symmetry (dodecagonal geometry)
- Periodic boundary conditions (toroidal topology)
- Atlantean mapping: φ₀→threshold, φ₄→leak
- Hybrid neural-field integration
- Diagnostic metrics (energy, coupling, symmetry)

---

## Quick Start

### Basic Field Evolution

```python
from quadra.core.field import FieldEngine

# Initialize 10-field substrate on 32×32 grid
engine = FieldEngine(shape=(32, 32), mode="2d", eps=0.15)

# Evolve fields forward
for step in range(100):
    engine.step(dt=0.01)
    
# Access fields and potential
phi_0 = engine.phi[0]      # First field (excitability)
Phi = engine.Phi            # Global potential
```

### Adaptive Control Parameters

```python
# Get spatially-varying thresholds
threshold_map = engine.dynamic_threshold(base=1.0)
# → base - 0.5·φ₀

# Get spatially-varying leak rates  
leak_map = engine.effective_leak(base=0.1)
# → base/(1 + exp(φ₄))
```

### Hybrid Neural-Field System

```python
from quadra.core.field import IFTEnhancedOscillator

# Create hybrid system
hybrid = IFTEnhancedOscillator(
    field_shape=(32, 32),
    neural_size=100,
    eps=0.12
)

# Co-evolve fields and neurons
neural_activity = get_neural_spikes()  # Your neural dynamics
hybrid.step_coupled(neural_activity, dt=0.01)

# Extract field-modulated parameters
thresholds = hybrid.get_neural_thresholds()  # → [neural_size]
leaks = hybrid.get_neural_leaks()            # → [neural_size]
```

---

## Field Specifications

### Field Assignments

| Field | Current Use | Formula |
|-------|-------------|----------|
| φ₀ | Threshold modulation | θ = base - 0.5·mean(φ₀) |
| φ₁ | Unused | N/A |
| φ₂ | Unused | N/A |
| φ₃ | Unused | N/A |
| φ₄ | Leak modulation | λ = base/(1+exp(mean(φ₄))) |
| φ₅ | Unused | N/A |
| φ₆ | Learning rate factor | η = 0.5+1/(1+exp(-mean(φ₆))) |
| φ₇ | Unused | N/A |
| φ₈ | Unused | N/A |
| φ₉ | Unused | N/A |

### Evolution Equations

Each field evolves via:

```
∂φₙ/∂t = Fₙ + ε·coupling_n

where Fₙ = -γₙ·φₙ³             # Cubic damping
           -αₙ·|φₙ|²·φₙ         # Saturation
           +βₙ·∇⁴φₙ             # Elastic stiffness
           -0.05·∇²φₙ           # Diffusion
```

**Parameters** (per field):
- γₙ = 0.1 (damping coefficient)
- αₙ = 0.2 (saturation coefficient)
- βₙ = 0.05 (stiffness coefficient)

### Resonant Coupling

The coupling tensor implements 12-fold symmetry:

```
coupling_n(x) = ε · cos(12θ(x)) · Σ_{m≠n} φₘ(x)
```

where:
- θ(x) = arctan2(y-y_center, x-x_center) is angular position
- ε = 0.15 is coupling strength
- 12-fold pattern creates dodecagonal symmetry

---

## Parameter Modulation

### Threshold Calculation

```python
θ(x) = base - 0.5·φ₀(x)
```

**Behavior:**
- Positive φ₀ values decrease threshold
- Negative φ₀ values increase threshold

**Usage:**
```python
threshold = engine.dynamic_threshold(base=1.0)
# Use in spiking model: spike when V > threshold
```

### Leak Rate Calculation

```python
λ(x) = base · 1/(1 + exp(φ₄(x)))
```

**Behavior:**
- Positive φ₄ values decrease leak rate (sigmoid function)
- Negative φ₄ values increase leak rate

**Usage:**
```python
leak = engine.effective_leak(base=0.1)
# Use in membrane dynamics: dV/dt = ... - λ·V
```

---

## Architecture Integration

### With Existing Oscillator

```python
from atlantean_hrm_spi import OscillatorySynapseTheory
from quadra.core.field import FieldEngine

# Standard oscillator
oscillator = OscillatorySynapseTheory(field_size=100)

# IFT substrate
ift = FieldEngine(shape=(32, 32), eps=0.15)

# Coupling loop
for step in range(1000):
    # Get adaptive parameters from IFT
    threshold_map = ift.dynamic_threshold()
    
    # Sample for neurons (spatial→neural mapping)
    # ... your mapping logic ...
    
    # Run neural dynamics with modulated params
    # ... oscillator.step() ...
    
    # Evolve IFT
    ift.step(dt=0.01)
```

### With Memory Store

```python
from quadra.state.memory_store import MemoryStore
from quadra.core.field import FieldEngine

memory = MemoryStore()
ift = FieldEngine(shape=(32, 32))

# Store field state
memory.core_field = ift.phi[0].flatten()  # Store φ₀

# Track field metrics
ift.step()
energy = ift.get_field_energy()
memory.syntropy_values.append(1.0 / (1.0 + energy))  # Lower energy → higher order
```

---

## Diagnostic Tools

### Energy Tracking

```python
energy = engine.get_field_energy()
# E = Σₙ ∫|φₙ|² dx
# Monitors total field energy (stability check)
```

### Coupling Strength

```python
coupling_01 = engine.get_coupling_strength(0, 1)
# C_nm = ε·<φₙ·φₘ> (spatial correlation)
# Measures effective interaction between fields
```

### Symmetry Order

```python
symmetry = engine.get_symmetry_order()
# Projects Φ onto cos(12θ) mode
# Returns [0,1]: how well field honors 12-fold geometry
```

---

## Modes

### 2D Mode (Default)

```python
engine = FieldEngine(shape=(H, W), mode="2d")
```

**Features:**
- Spatial fields on HxW grid
- Laplacian and bi-Laplacian operators
- Periodic boundary conditions (torus topology)
- 12-fold angular symmetry in coupling

**Use for:** Spatially-extended systems, pattern formation, wave propagation

### 0D Mode

```python
engine = FieldEngine(shape=(N,), mode="0d")
```

**Features:**
- Vector fields (no spatial structure)
- No Laplacian terms (lap=0, bilap=0)
- Faster computation
- Still includes resonant coupling

**Use for:** Non-spatial dynamics, faster prototyping, scalar systems

---

## Parameter Tuning

### Coupling Strength (ε)

```python
engine = FieldEngine(eps=0.15)  # Default
```

- **Low ε** (0.05-0.10): Weakly coupled, fields evolve independently
- **Medium ε** (0.10-0.20): Moderate coupling, stable patterns
- **High ε** (0.20-0.30): Strong coupling, complex dynamics (may destabilize)

### Resonance Frequency (ω₀)

```python
engine = FieldEngine(omega0=1.4e14)  # UV range, ~2 PHz
```

Currently a placeholder for future frequency-selective coupling. Could implement:
```python
gate = np.exp(-np.abs(omega - omega0) / Gamma)  # Lorentzian resonance
```

### Field-Specific Parameters

```python
engine.gamma[0] = 0.2  # Increase φ₀ damping
engine.alpha[5] = 0.1  # Decrease φ₅ saturation
engine.beta[3] = 0.1   # Increase φ₃ stiffness
```

**γₙ (damping):** Higher → faster decay, more stable  
**αₙ (saturation):** Higher → stronger nonlinearity  
**βₙ (stiffness):** Higher → smoother spatial patterns

---

## Examples

### Complete Demo

Run the full demonstration:

```bash
python demo_ift_fields.py
```

**Output:**
- Evolution metrics (energy, symmetry)
- Adaptive parameter ranges
- Coupling matrix
- Hybrid neural-field dynamics
- Visualization (4-panel plot)

### Custom Evolution

```python
import numpy as np
from quadra.core.field import FieldEngine

engine = FieldEngine(shape=(64, 64), eps=0.18)

# Custom initial condition: Gaussian bump
H, W = engine.shape
yy, xx = np.mgrid[0:H, 0:W]
r2 = (xx - H/2)**2 + (yy - W/2)**2
engine.phi[0] = np.exp(-r2 / 100)

# Evolve and track
for step in range(500):
    engine.step(dt=0.01)
    
    if step % 50 == 0:
        symmetry = engine.get_symmetry_order()
        print(f"Step {step}: 12-fold symmetry = {symmetry:.4f}")
```

### Pattern Formation

```python
import matplotlib.pyplot as plt

engine = FieldEngine(shape=(128, 128), eps=0.20)

# Evolve to steady state
for _ in range(1000):
    engine.step(dt=0.01)

# Visualize all fields
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for n in range(10):
    ax = axes[n // 5, n % 5]
    im = ax.imshow(engine.phi[n], cmap='RdBu_r')
    ax.set_title(f'φ{n}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

plt.suptitle('Ten-Field Pattern Formation')
plt.tight_layout()
plt.savefig('field_patterns.png', dpi=150)
```

---

## Performance

### Computational Cost

**Per time step:**
- 2D mode: O(10 × H × W) for field updates + O(H × W) for Laplacians
- 0D mode: O(10 × N) (faster, no spatial operators)

**Typical timing** (on CPU):
- 32×32 grid: ~1 ms/step
- 64×64 grid: ~4 ms/step
- 128×128 grid: ~15 ms/step

### Optimization Tips

1. **Use 0D mode** if spatial structure not needed
2. **Reduce grid size** for prototyping (16×16 is often sufficient)
3. **Vectorize operations** (already done in implementation)
4. **GPU acceleration** (future: CuPy/JAX backend)

---

## Implementation Notes

### Coupling Pattern

The cos(12θ) coupling term creates spatial variation based on angular position in the grid. The value 12 was chosen as a parameter.

### Parameter Modulation

Field averages are used to modulate three system parameters:
- φ₀ mean modulates threshold via linear function
- φ₄ mean modulates leak rate via sigmoid function  
- φ₆ mean modulates learning rate via sigmoid function

---

## Future Enhancements

### Planned Features

- [ ] **Learnable parameters:** Train γₙ, αₙ, βₙ via gradient descent
- [ ] **Frequency-selective coupling:** Implement ω-dependent resonance gates
- [ ] **3D mode:** Volumetric fields for full spatial embedding
- [ ] **GPU backend:** CuPy/JAX for 10-100x speedup
- [ ] **Adaptive symmetry:** Dynamic angular order (not fixed at 12)
- [ ] **Multi-scale:** Hierarchical field systems

### Research Directions

- Stability analysis of coupled PDEs
- Bifurcation theory for pattern transitions
- Information-theoretic measures of field coupling
- Experimental validation with neural data

---

## References

**Field Theory:**
- Ramond, P. (2001). *Field Theory: A Modern Primer*
- Peskin & Schroeder (1995). *Introduction to Quantum Field Theory*

**Pattern Formation:**
- Cross & Hohenberg (1993). "Pattern formation outside of equilibrium"
- Murray, J.D. (2002). *Mathematical Biology*

**Neuroscience:**
- Buzsáki, G. (2006). *Rhythms of the Brain*
- Freeman, W.J. (2000). *How Brains Make Up Their Minds*

---

## Contact

For questions, contributions, or discussion:
- **GitHub Issues:** [cognitionsim/cognitionsim](https://github.com/cognitionsim/cognitionsim/issues)
- **Documentation:** See `DUAL_LANGUAGE_GLOSSARY.md` for concept translations

---
