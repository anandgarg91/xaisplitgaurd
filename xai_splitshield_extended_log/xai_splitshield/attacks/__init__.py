"""attacks/__init__.py"""
from .badnets import BadNetsAttack
from .blended import BlendedAttack
from .wanet import WaNetAttack
from .lira import LIRAAttack
__all__ = ["BadNetsAttack", "BlendedAttack", "WaNetAttack", "LIRAAttack"]
