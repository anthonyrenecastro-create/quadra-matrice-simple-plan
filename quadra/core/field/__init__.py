"""
Field Dynamics Module - Substrate-Level Physics

🌊 POETIC:
The foundation upon which all emerges—ten cosmic fields in resonant harmony,
creating the substrate for consciousness through sacred geometry and adaptive
self-regulation.

⚙️ MECHANICAL:
Implements field-theoretic foundations for CognitionSim:
- Interplayed Field Theory (IFT) ten-field substrate
- 12-fold angular symmetry via resonant tensors
- Atlantean control: adaptive thresholds and leak rates
- Hybrid IFT-neural integration

See: ../../DUAL_LANGUAGE_GLOSSARY.md for concept translations
"""

from .ift_engine import (
    FieldEngine,
    IFTEnhancedOscillator
)
from .syntropy_field import SyntropyField
from .field_nn import FieldUpdateNN, SyntropyNN, FeedbackNN
from .chaos import ChaosController
from .governance import Governance

__all__ = [
    'FieldEngine',
    'IFTEnhancedOscillator',
    'SyntropyField',
    'FieldUpdateNN',
    'SyntropyNN',
    'FeedbackNN',
    'ChaosController',
    'Governance',
]
