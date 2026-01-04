"""
Models Module

Contém modelos de ML e sistemas de ensemble.
"""

from .ensemble_model import EnsembleModel, VotingStrategy

__all__ = ['EnsembleModel', 'VotingStrategy']
