
# from __future__ import absolute_import

__version__ = "0.1"

from .smartdoc_loader  import (load_sd15ch1_frames, load_sd15ch1_models,
    MODEL_VARIANT_01_ORIGINAL, MODEL_VARIANT_02_EDITED, MODEL_VARIANT_03_CAPTURED,
    MODEL_VARIANT_04_CORRECTED, MODEL_VARIANT_05_SCALED33)

__all__ = [
    'load_sd15ch1_frames',
    'load_sd15ch1_models',
    'MODEL_VARIANT_01_ORIGINAL',
    'MODEL_VARIANT_02_EDITED',
    'MODEL_VARIANT_03_CAPTURED',
    'MODEL_VARIANT_04_CORRECTED',
    'MODEL_VARIANT_05_SCALED33',
    ]
