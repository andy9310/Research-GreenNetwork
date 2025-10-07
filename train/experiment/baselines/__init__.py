"""
Baseline algorithms for comparison
"""

from .energy_aware_routing import EnergyAwareRouting
from .rl_energy_routing import RLEnergyRouting

__all__ = ['EnergyAwareRouting', 'RLEnergyRouting']
