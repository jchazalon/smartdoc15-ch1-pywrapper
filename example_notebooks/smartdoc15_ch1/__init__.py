

# This is the place where the version is defined.
__version__ = "0.8"


from .smartdoc_loader_v2 import Dataset, Models
from .evaluation import evaluate_segmentation, evaluate_classification

from .smartdoc_loader  import (load_sd15ch1_frames, load_sd15ch1_models,
    MODEL_VARIANT_01_ORIGINAL, MODEL_VARIANT_02_EDITED, MODEL_VARIANT_03_CAPTURED,
    MODEL_VARIANT_04_CORRECTED, MODEL_VARIANT_05_SCALED33, eval_sd15ch1_segmentations,
    eval_sd15ch1_classifications, get_sd15ch1_basedir_frames, get_sd15ch1_basedir_models)

__all__ = [
    # V2
    'Dataset',
    'Models',
    'evaluate_segmentation',
    'evaluate_classification',
    # V1
    'load_sd15ch1_frames',
    'load_sd15ch1_frames',
    'load_sd15ch1_models',
    'read_sd15ch1_image',
    'read_sd15ch1_images',
    'MODEL_VARIANT_01_ORIGINAL',
    'MODEL_VARIANT_02_EDITED',
    'MODEL_VARIANT_03_CAPTURED',
    'MODEL_VARIANT_04_CORRECTED',
    'MODEL_VARIANT_05_SCALED33',
    'eval_sd15ch1_segmentations',
    'eval_sd15ch1_classifications',
    'get_sd15ch1_basedir_frames',
    'get_sd15ch1_basedir_models',
    ]
