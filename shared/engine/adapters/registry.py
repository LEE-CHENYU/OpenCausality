"""
Adapter Registry.

Config-driven registry that reads `adapter_class` from design_registry.yaml
and dynamically imports the adapter class.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from shared.engine.adapters.base import EstimatorAdapter

# Built-in adapter mapping (design_id -> module.ClassName)
_BUILTIN_ADAPTERS: dict[str, str] = {
    "LOCAL_PROJECTIONS": "shared.engine.adapters.lp_adapter.LPAdapter",
    "PANEL_LP_EXPOSURE_FE": "shared.engine.adapters.panel_lp_adapter.PanelLPAdapter",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "config" / "agentic" / "design_registry.yaml"

# Cache loaded adapters
_adapter_cache: dict[str, EstimatorAdapter] = {}


def _load_registry_adapters() -> dict[str, str]:
    """Load adapter_class mappings from design_registry.yaml."""
    mapping: dict[str, str] = {}
    if not _REGISTRY_PATH.exists():
        return mapping
    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f) or {}
    for design in data.get("designs", []):
        design_id = design.get("id", "")
        adapter_class = design.get("adapter_class", "")
        if design_id and adapter_class:
            mapping[design_id] = adapter_class
    return mapping


def _import_adapter(dotted_path: str) -> type[EstimatorAdapter]:
    """Dynamically import an adapter class from a dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, EstimatorAdapter)):
        raise TypeError(f"{dotted_path} is not an EstimatorAdapter subclass")
    return cls


def get_adapter(design_id: str) -> EstimatorAdapter:
    """Get an adapter instance for a design ID.

    Lookup order:
    1. Config-driven: adapter_class field in design_registry.yaml
    2. Built-in mapping

    Args:
        design_id: Design identifier (e.g., "LOCAL_PROJECTIONS").

    Returns:
        An EstimatorAdapter instance.

    Raises:
        ValueError: If no adapter is registered for the design.
    """
    if design_id in _adapter_cache:
        return _adapter_cache[design_id]

    # Try config-driven first
    config_adapters = _load_registry_adapters()
    dotted_path = config_adapters.get(design_id) or _BUILTIN_ADAPTERS.get(design_id)

    if not dotted_path:
        raise ValueError(
            f"No adapter registered for design '{design_id}'. "
            f"Available: {sorted(set(list(config_adapters.keys()) + list(_BUILTIN_ADAPTERS.keys())))}"
        )

    cls = _import_adapter(dotted_path)
    instance = cls()
    _adapter_cache[design_id] = instance
    return instance


def list_adapters() -> dict[str, str]:
    """List all registered adapter mappings (design_id -> class path)."""
    config_adapters = _load_registry_adapters()
    merged = dict(_BUILTIN_ADAPTERS)
    merged.update(config_adapters)
    return merged
