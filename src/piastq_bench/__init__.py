"""PIAST-Q emulator benchmark for EuroHPC project EHPC-QCP-2026Q04-092."""

__version__ = "0.2.0"

from .core import (
    anneal,
    anneal_3body,
    energy_table,
    energy_table_3body,
    make_3body_instance,
    make_density_instance,
    make_instance,
    qaoa_state,
    sample_metrics,
    select_qaoa_angles,
    valid_mask,
    valid_mask_3body,
)

__all__ = [
    "__version__",
    "anneal",
    "anneal_3body",
    "energy_table",
    "energy_table_3body",
    "make_3body_instance",
    "make_density_instance",
    "make_instance",
    "qaoa_state",
    "sample_metrics",
    "select_qaoa_angles",
    "valid_mask",
    "valid_mask_3body",
]
