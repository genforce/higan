# python 3.7
"""Contains basic configurations for predictors used in this project.

Please download the public released models and annotations from the following
repositories OR train your own predictor.

places365: https://github.com/CSAILVision/places365

NOTE: Any new predictor should be registered in `PREDICTOR_POOL` before used.
"""

import os.path

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

ANNO_DIR = os.path.join(BASE_DIR, 'annotations')
MODEL_DIR = os.path.join(BASE_DIR, 'pretrain')

# pylint: disable=line-too-long
PREDICTOR_POOL = {
    # Scene Predictor.
    'scene': {
        'weight_path': os.path.join(MODEL_DIR, 'wideresnet18_places365.pth.tar'),
        'resolution': None,  # Image resize will be done automatically.
        'image_channels': 3,
        'channel_order': 'RGB',
        'category_anno_path': os.path.join(ANNO_DIR, 'categories_places365.txt'),
        'attribute_anno_path': os.path.join(ANNO_DIR, 'labels_sunattribute.txt'),
        'attribute_additional_weight_path': os.path.join(MODEL_DIR, 'W_sceneattribute_wideresnet18.npy'),
    }
}
# pylint: enable=line-too-long

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 4

MAX_IMAGES_ON_RAM = 800
