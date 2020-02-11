# Semantic Preditors

All customized semantic predictors can be placed under this folder, derived from `base_predictor.py` and registered in `predictor_settings.py`. In this repo, scene-related predictor is provided.

## Scene Attribute and Category Prediction Model

This model is borrowed from [places365](https://github.com/CSAILVision/places365) repository.

Before using this predictor, please download following files:

- Definition of 365 categories on Places365 dataset. Please save file [categories_places365.txt](https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt) under `annotations` folder.
- Definition of 102 attributes on Places365 dataset. Please save file [labels_sunattribute.txt](https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt) under `annotations` folder.
- Pre-trained weight for category prediction. Please save file [wideresnet18_places365.pth.tar](http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar) under `pretrain` folder.
- Additional weight for attribute prediction. Please save file [W_sceneattribute_wideresnet18.npy](http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy) under `pretrain` folder.
- Existing file `scene_wideresnet.py` in this folder is borrowed from [here](https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py), yet slightly changed (Line 152 and Line 156).
