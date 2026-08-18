"""Small compatibility boundary around the vendored CON-FOLD runtime.

This module only constructs CONFOLD classifiers and contains vendor-specific
import and schema quirks. The decision-dataset extractor owns feature and label
construction; ``run_confold_baseline.py`` owns its offline leave-one-run-out
split and evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


CONFOLD_ROOT = Path(__file__).resolve().parent / "confold"


def make_classifier(
    features: Sequence[str],
    numeric_features: Sequence[str],
    label: str,
):
    """Create a fresh CONFOLD model while keeping vendor quirks contained.

    The upstream project uses older flat-module imports, so its directory is
    made importable here rather than throughout AP4Fed.  Its CSV loader can
    append the label to the supplied attribute list, so the schema lists are
    copied even though this runner trains from in-memory rows.
    """
    vendor_path = str(CONFOLD_ROOT)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    try:
        from foldrm import Classifier
    except ModuleNotFoundError as exc:
        if exc.name == "scipy":
            raise RuntimeError(
                "CON-FOLD requires SciPy. Install distill/requirements-confold.txt."
            ) from exc
        raise
    return Classifier(attrs=list(features), numeric=list(numeric_features), label=label)
