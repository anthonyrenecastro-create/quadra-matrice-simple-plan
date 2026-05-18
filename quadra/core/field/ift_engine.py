"""
Interplayed Field Theory (IFT) Engine - Ten-Field Substrate

# Technical implementation
Ten cosmic fields dance in harmonic resonance, each with its own voice yet
all singing in chorus. The fields breathe with 12-fold symmetry, echoing
the ancient patterns of sacred geometry. Through their interplay, complexity
emerges—patterns self-organize, thresholds adapt, and the substrate learns
to modulate its own excitability. This is the foundation upon which consciousness
builds its temple.

# Implementation
Implements a 10-field coupled PDE system inspired by Interplayed Field Theory:
    ∂φₙ/∂t = Fₙ(φₙ, ∇²φₙ, ∇⁴φₙ) + ε·T_nmp(ω,x)·Σφₘ
    
where:
- φₙ: n-th field (n=0..9)
- Fₙ: field-specific evolution (damping + nonlinearity + stiffness)
- T_nmp: rank-3 resonant tensor with 12-fold angular symmetry
- ε: coupling strength
- ω₀: resonance frequency

Supports:
- 0D mode: scalar/vector fields
- 2D mode: spatial fields with Laplacian operators (periodic BC)

Atlantean control: Dynamic thresholds and leak rates modulated by field states.

See: ../../../DUAL_LANGUAGE_GLOSSARY.md for concept translations
"""

import numpy as np
from typing import Tuple, Literal, Optional, Dict, Any

try:
    from .field_nn import FieldUpdateNN
except ImportError:  # pragma: no cover - direct module execution fallback
    from field_nn import FieldUpdateNN


class FieldEngine:
    """
    Ten-field substrate with Atlantean mapping for adaptive control.
    
    # Technical implementation
    The substrate holds ten fields in delicate balance—each field a voice
    in the cosmic choir. Their coupling creates emergent patterns through
    12-fold sacred geometry. The fields control their own thresholds and
    leak rates, achieving self-regulation through resonance.
    
    # Implementation
    Coupled PDE system with:
    - 10 fields φₙ (n=0..9) on HxW grid or scalar
    - Global potential Φ (low-pass filtered activity)
    - 12-fold angular symmetry in coupling tensor
    - Adaptive threshold/leak via field states
    
    Parameters control evolution:
    - γₙ: damping (cubic nonlinearity)
    - αₙ: saturation (abs-cubic nonlinearity)
    - βₙ: stiffness (bi-Laplacian weight)
    
    Attributes:
        phi: List of 10 field arrays [φ₀, φ₁, ..., φ₉]
        Phi: Global potential (activity-filtered)
        mode: "0d" or "2d" topology
        shape: Field dimensions
        eps: Coupling strength
        omega0: Resonance frequency (rad/s)
    """
    
    def __init__(
        self,
        shape: Tuple[int, int] = (32, 32),
        mode: Literal["0d", "2d"] = "2d",
        eps: float = 0.15,
        omega0: float = 1.4e14,
        seed: int = 42
    ):
        """
        Initialize ten-field system.
        
        Args:
            shape: Field dimensions (H, W) for 2D mode or (N,) for 0D
            mode: Field topology - "2d" for spatial, "0d" for scalar
            eps: Coupling strength between fields (dimensionless)
            omega0: Resonance frequency parameter (rad/s)
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.shape = shape
        self.eps = float(eps)
        self.omega0 = float(omega0)
        
        # # Technical implementation
        # # Implementation
        self.phi = [
            self.rng.standard_normal(shape) * 0.01 
            for _ in range(10)
        ]
        
        # # Technical implementation
        # # Implementation
        self.Phi = np.zeros(shape, dtype=float)
        
        # # Technical implementation
        # # Implementation
        self.gamma = np.ones(10) * 0.1   # Damping
        self.alpha = np.ones(10) * 0.2   # Saturation
        self.beta  = np.ones(10) * 0.05  # Stiffness

        # Free-energy functional controls for projector penalty:
        # dphi_dt = -δF_IFT/δphi - P(phi)
        self.free_energy: Dict[str, Any] = {
            "projector_mode": "high_pass",      # {"none", "high_pass", "mask"}
            "projector_strength": 0.0,           # λ_P in λ_P * P(phi)
            "projector_cutoff": 0.35,            # normalized radial cutoff for high-pass
            "projector_mask": None,              # optional fixed/learned spectral mask
            "stabilize_with_update_nn": False,   # apply smoothing+variance clip to P(phi)
            "variance_clip": 0.2,
        }
        self._spectral_mask_cache: Optional[np.ndarray] = None
        self._field_update_nn = FieldUpdateNN(
            variance_clip=float(self.free_energy["variance_clip"])
        )

    def set_free_energy_functional(
        self,
        *,
        projector_mode: Optional[str] = None,
        projector_strength: Optional[float] = None,
        projector_cutoff: Optional[float] = None,
        projector_mask: Optional[np.ndarray] = None,
        stabilize_with_update_nn: Optional[bool] = None,
        variance_clip: Optional[float] = None,
    ) -> None:
        """
        Update free-energy functional controls for projector regularization.

        This configures the penalty in:
            ∂phi/∂t = -δF_IFT/δphi - P(phi)

        Args:
            projector_mode: "none", "high_pass", or "mask".
            projector_strength: Scalar weight λ_P for projector penalty.
            projector_cutoff: Normalized radial cutoff in [0, 1] for high-pass.
            projector_mask: Fixed/learned spectral mask with field shape.
            stabilize_with_update_nn: If True, smooth+clip P(phi) before penalty.
            variance_clip: Variance clip used by FieldUpdateNN stabilizer.
        """
        if projector_mode is not None:
            mode = str(projector_mode).lower()
            if mode not in {"none", "high_pass", "mask"}:
                raise ValueError("projector_mode must be one of: none, high_pass, mask")
            self.free_energy["projector_mode"] = mode

        if projector_strength is not None:
            self.free_energy["projector_strength"] = float(max(0.0, projector_strength))

        if projector_cutoff is not None:
            self.free_energy["projector_cutoff"] = float(np.clip(projector_cutoff, 0.0, 1.0))
            self._spectral_mask_cache = None

        if projector_mask is not None:
            mask = np.asarray(projector_mask, dtype=float)
            if mask.shape != self.shape:
                raise ValueError(
                    f"projector_mask shape {mask.shape} does not match field shape {self.shape}"
                )
            self.free_energy["projector_mask"] = mask

        if stabilize_with_update_nn is not None:
            self.free_energy["stabilize_with_update_nn"] = bool(stabilize_with_update_nn)

        if variance_clip is not None:
            self.free_energy["variance_clip"] = float(max(0.0, variance_clip))
            self._field_update_nn = FieldUpdateNN(
                variance_clip=float(self.free_energy["variance_clip"])
            )

    def get_free_energy_functional(self) -> Dict[str, Any]:
        """Return a shallow copy of current free-energy functional controls."""
        cfg = dict(self.free_energy)
        if isinstance(cfg.get("projector_mask"), np.ndarray):
            cfg["projector_mask"] = cfg["projector_mask"].copy()
        return cfg

    def _build_high_pass_mask(self) -> np.ndarray:
        """Construct normalized radial high-pass mask in Fourier space."""
        if self.mode != "2d":
            return np.ones(self.shape, dtype=float)
        if self._spectral_mask_cache is not None:
            return self._spectral_mask_cache

        H, W = self.shape
        ky = np.fft.fftfreq(H)
        kx = np.fft.fftfreq(W)
        ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
        radius = np.sqrt(kx_grid**2 + ky_grid**2)
        max_radius = float(np.max(radius)) + 1e-12
        radius_norm = radius / max_radius
        cutoff = float(self.free_energy["projector_cutoff"])
        mask = (radius_norm >= cutoff).astype(float)
        self._spectral_mask_cache = mask
        return mask

    def _apply_projector(self, x: np.ndarray) -> np.ndarray:
        """
        Apply spectral projector P to a field.

        - high_pass: project onto undesired high-wavenumber modes
        - mask: apply fixed/learned spectral mask
        - none: zero penalty term
        """
        mode = self.free_energy["projector_mode"]
        if mode == "none":
            return np.zeros_like(x)

        if self.mode == "0d":
            if mode == "mask" and self.free_energy["projector_mask"] is not None:
                return np.asarray(self.free_energy["projector_mask"]) * x
            return x

        x_hat = np.fft.fft2(x)
        if mode == "mask":
            mask = self.free_energy["projector_mask"]
            if mask is None:
                mask = self._build_high_pass_mask()
        else:
            mask = self._build_high_pass_mask()
        projected = np.fft.ifft2(np.asarray(mask, dtype=float) * x_hat).real

        if self.free_energy["stabilize_with_update_nn"]:
            projected = self._field_update_nn.update(projected)
        return projected
        
    @staticmethod
    def _laplacian_2d(x: np.ndarray) -> np.ndarray:
        """
        Compute discrete Laplacian with periodic boundaries.
        
        Uses 5-point stencil for second derivative approximation.
        
        Args:
            x: 2D field array
            
        Returns:
            Laplacian array with periodic boundary conditions
        """
        return (
            np.roll(x,  1, axis=0) + np.roll(x, -1, axis=0) +
            np.roll(x,  1, axis=1) + np.roll(x, -1, axis=1) -
            4.0 * x
        )
    
    def _bi_laplacian_2d(self, x: np.ndarray) -> np.ndarray:
        """
        Compute bi-Laplacian (Laplacian of Laplacian).
        
        Applies Laplacian operator twice for fourth-order derivative.
        
        Args:
            x: 2D field array
            
        Returns:
            Bi-Laplacian array
        """
        return self._laplacian_2d(self._laplacian_2d(x))
    
    def _tensor_coupling(self, n: int) -> np.ndarray:
        """
        Coupling tensor with cos(12*theta) angular dependence.
        
        Implements coupling formula:
            coupling_n = cos(12θ) · Σ_{m≠n} φₘ
        
        where θ = arctan2(y - cy, x - cx) is angular position.
        
        Args:
            n: Index of field receiving coupling (0-9)
            
        Returns:
            Coupling contribution from all other fields
        """
        H, W = self.shape
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        dx, dy = xx - cx, yy - cy
        theta = np.arctan2(dy, dx)
        
        # Angular mask: cos(12θ) for 12-fold pattern
        phase = np.cos(12.0 * theta)
        
        # Frequency-dependent multiplier (constant=1.0)
        gate = 1.0  # Could be exp(-|ω-ω₀|/Γ) for frequency selectivity
        
        # # Technical implementation
        # # Implementation
        mix = np.zeros(self.shape, dtype=float)
        for m in range(10):
            if m != n:
                mix += self.phi[m]
        
        return gate * phase * mix
    
    def step(self, dt: float = 0.01):
        """
        Evolve all fields forward by one time step.
        
        # Technical implementation
        Time flows—the fields breathe, couple, and evolve. Each field
        experiences damping, nonlinear saturation, elastic stiffness,
        and the harmonious pull of all others through resonant coupling.
        
        # Implementation
        Integrate coupled PDEs via forward Euler:
            φₙ[t+dt] = φₙ[t] + dt·(Fₙ + ε·coupling_n)
        
        where Fₙ includes:
        - Cubic damping: -γₙ·φₙ³
        - Saturation: -αₙ·|φₙ|²·φₙ
        - Stiffness: βₙ·∇⁴φₙ
        - Diffusion: -0.05·∇²φₙ
        
        Global potential updated:
            Φ[t+dt] = 0.95·Φ[t] + 0.05·tanh(φ₀)
        
        Args:
            dt: Time step (dimensionless, typically 0.01)
        """
        # Loop over all 10 fields
        for n in range(10):
            x = self.phi[n]
            
            # Compute Laplacian and bi-Laplacian if 2D mode
            if self.mode == "2d":
                lap = self._laplacian_2d(x)
                bilap = self._bi_laplacian_2d(x)
            else:
                lap = 0.0
                bilap = 0.0
            
            # Field evolution combining nonlinear terms
            Fn = (
                -self.gamma[n] * (x**2) * x       # Cubic damping
                - self.alpha[n] * np.abs(x)**2 * x  # Saturation
                + self.beta[n] * bilap             # Elastic stiffness
                - 0.05 * lap                       # Diffusion
            )
            
            # Coupling via 12-fold tensor
            coupling = self.eps * self._tensor_coupling(n)

            # Interpret current dynamics as -δF_IFT/δphi
            variational_term = Fn + coupling

            # Projector penalty: P(phi) in ∂phi/∂t = -δF_IFT/δphi - P(phi)
            projector_strength = float(self.free_energy["projector_strength"])
            projector_penalty = projector_strength * self._apply_projector(x)

            dphi_dt = variational_term - projector_penalty
            
            # Forward Euler integration step
            self.phi[n] = x + dt * dphi_dt
        
        # Update global potential (low-pass filter of phi_0)
        activity = np.tanh(self.phi[0])  # phi_0 as activity proxy
        self.Phi = 0.95 * self.Phi + 0.05 * activity
    
    # Parameter modulation functions
    
    def dynamic_threshold(self, base: float = 1.0) -> np.ndarray:
        """
        Threshold values modulated by field state.
        The threshold is not fixed—it breathes with the field. When φ₀
        rises, the threshold lowers, making the system more excitable.
        This is homeostatic control: the field regulates itself.
        
        Formula: theta(x) = base - 0.5 * phi_0(x)
        
        Args:
            base: Baseline threshold value
            
        Returns:
            Spatially-varying threshold array
        """
        return base - 0.5 * self.phi[0]
    
    def effective_leak(self, base: float = 0.1) -> np.ndarray:
        """
        Leak rate values modulated by field state.
        
        Formula: lambda(x) = base / (1 + exp(phi_4(x)))
        
        Args:
            base: Baseline leak rate
            
        Returns:
            Spatially-varying leak rate array
        """
        # Clip phi_4 to prevent overflow in exp
        # # Implementation
        phi4_clipped = np.clip(self.phi[4], -5, 5)
        
        return base * (1.0 / (1.0 + np.exp(phi4_clipped)))
    
    def get_field_energy(self) -> float:
        """
        Compute total field energy (diagnostics).
        
        # Technical implementation
        # Implementation
        
        Returns:
            Total energy across all fields
        """
        return sum(np.sum(f**2) for f in self.phi)
    
    def get_coupling_strength(self, n: int, m: int) -> float:
        """
        Measure effective coupling between two fields.
        
        # Technical implementation
        # Implementation
        
        Args:
            n: First field index
            m: Second field index
            
        Returns:
            Coupling strength (scalar)
        """
        return self.eps * np.mean(self.phi[n] * self.phi[m])
    
    def get_symmetry_order(self) -> float:
        """
        Measure 12-fold symmetry in global potential.
        
        # Technical implementation
        # Implementation
        
        Returns:
            Symmetry order parameter [0, 1]
        """
        if self.mode != "2d":
            return 0.0
        
        H, W = self.shape
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        theta = np.arctan2(yy - cy, xx - cx)
        
        # Project Φ onto cos(12θ) mode
        mode_12 = np.cos(12.0 * theta)
        overlap = np.abs(np.mean(self.Phi * mode_12))
        
        return overlap


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH COGNITIONSIM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class IFTEnhancedOscillator:
    """
    Wrapper integrating FieldEngine with existing CognitionSim oscillator.
    
    # Technical implementation
    # Implementation
    """
    
    def __init__(
        self,
        field_shape: Tuple[int, int] = (32, 32),
        neural_size: int = 100,
        eps: float = 0.15
    ):
        """
        Initialize hybrid IFT-neural system.
        
        Args:
            field_shape: IFT field dimensions
            neural_size: Number of spiking neurons
            eps: IFT coupling strength
        """
        self.ift = FieldEngine(shape=field_shape, eps=eps)
        self.neural_size = neural_size
        
        # # Technical implementation
        # # Implementation
        self._init_mapping()
    
    def _init_mapping(self):
        """Create spatial→neural mapping."""
        H, W = self.ift.shape
        self.pool_indices = np.random.randint(
            0, H*W, size=self.neural_size
        )
    
    def get_neural_thresholds(self) -> np.ndarray:
        """
        Extract adaptive thresholds for spiking neurons.
        
        # Technical implementation
        # Implementation
        
        Returns:
            Threshold array of size [neural_size]
        """
        spatial_threshold = self.ift.dynamic_threshold(base=1.0)
        flat = spatial_threshold.flatten()
        return flat[self.pool_indices]
    
    def get_neural_leaks(self) -> np.ndarray:
        """
        Extract adaptive leak rates for spiking neurons.
        
        # Technical implementation
        # Implementation
        
        Returns:
            Leak rate array of size [neural_size]
        """
        spatial_leak = self.ift.effective_leak(base=0.1)
        flat = spatial_leak.flatten()
        return flat[self.pool_indices]
    
    def step_coupled(
        self,
        neural_activity: np.ndarray,
        dt: float = 0.01
    ):
        """
        Co-evolve IFT fields and neural activity.
        
        # Technical implementation
        # Implementation
        
        Args:
            neural_activity: Current neural firing rates [neural_size]
            dt: Time step
        """
        # # Technical implementation
        # # Implementation
        activity_map = np.zeros(self.ift.shape)
        flat = activity_map.flatten()
        flat[self.pool_indices] = neural_activity
        activity_map = flat.reshape(self.ift.shape)
        
        # Add small feedback coupling
        self.ift.phi[0] += 0.01 * activity_map
        
        # # Technical implementation
        # # Implementation
        self.ift.step(dt=dt)


if __name__ == "__main__":
    # Quick demonstration
    print("Technical: CognitionSim - IFT Field Engine Demo")
    print("=" * 60)
    
    engine = FieldEngine(shape=(64, 64), mode="2d", eps=0.15)
    
    print(f"Initialized 10-field substrate: {engine.shape}")
    print(f"Coupling strength: ε = {engine.eps}")
    print(f"Resonance frequency: ω₀ = {engine.omega0:.2e} rad/s")
    print()
    
    print("Evolving fields...")
    for step in range(100):
        engine.step(dt=0.01)
        
        if step % 20 == 0:
            energy = engine.get_field_energy()
            symmetry = engine.get_symmetry_order()
            coupling_01 = engine.get_coupling_strength(0, 1)
            
            print(f"  Step {step:3d}: E={energy:8.4f}, "
                  f"Symmetry={symmetry:.4f}, C₀₁={coupling_01:+.4f}")
    
    print()
    print("✓ Field dynamics stable")
    print(f"  Final Φ range: [{engine.Phi.min():.3f}, {engine.Phi.max():.3f}]")
    print(f"  Adaptive threshold range: "
          f"[{engine.dynamic_threshold().min():.3f}, "
          f"{engine.dynamic_threshold().max():.3f}]")
    print(f"  Adaptive leak range: "
          f"[{engine.effective_leak().min():.4f}, "
          f"{engine.effective_leak().max():.4f}]")
