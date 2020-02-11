# python 3.7
"""Predicts the scene category, attribute."""

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .base_predictor import BasePredictor
from .scene_wideresnet import resnet18

__all__ = ['ScenePredictor']

NUM_CATEGORIES = 365
NUM_ATTRIBUTES = 102
FEATURE_DIM = 512

class ScenePredictor(BasePredictor):
  """Defines the predictor class for scene analysis."""

  def __init__(self):
    super().__init__('scene')

  def build(self):
    self.net = resnet18(num_classes=NUM_CATEGORIES)

  def load(self):
    # Load category labels.
    self.check_attr('category_anno_path')
    self.category_name_to_idx = {}
    self.category_idx_to_name = {}
    with open(self.category_anno_path, 'r') as f:
      for line in f:
        name, idx = line.strip().split(' ')
        name = name[3:].replace('/', '__')
        idx = int(idx)
        self.category_name_to_idx[name] = idx
        self.category_idx_to_name[idx] = name
    assert len(self.category_name_to_idx) == NUM_CATEGORIES
    assert len(self.category_idx_to_name) == NUM_CATEGORIES

    # Load attribute labels.
    self.check_attr('attribute_anno_path')
    self.attribute_name_to_idx = {}
    self.attribute_idx_to_name = {}
    with open(self.attribute_anno_path, 'r') as f:
      for idx, line in enumerate(f):
        name = line.strip().replace(' ', '_')
        self.attribute_name_to_idx[name] = idx
        self.attribute_idx_to_name[idx] = name
    assert len(self.attribute_name_to_idx) == NUM_ATTRIBUTES
    assert len(self.attribute_idx_to_name) == NUM_ATTRIBUTES

    # Transform for input images.
    self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load pre-trained weights for category prediction.
    checkpoint = torch.load(self.weight_path,
                            map_location=lambda storage, loc: storage)
    state_dict = {k.replace('module.', ''): v
                  for k, v in checkpoint['state_dict'].items()}
    self.net.load_state_dict(state_dict)
    fc_weight = list(self.net.parameters())[-2].data.numpy()
    fc_weight[fc_weight < 0] = 0

    # Load additional weights for attribute prediction.
    self.check_attr('attribute_additional_weight_path')
    self.attribute_weight = np.load(self.attribute_additional_weight_path)
    assert self.attribute_weight.shape == (NUM_ATTRIBUTES, FEATURE_DIM)

  def _predict(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')
    if images.dtype != np.uint8:
      raise ValueError(f'Images should be with dtype `numpy.uint8`!')
    if not (len(images.shape) == 4 and
            0 < images.shape[0] <= self.batch_size and
            images.shape[3] == self.image_channels):
      raise ValueError(f'Images should be with shape [batch_size, height '
                       f'width, channel], where `batch_size` no larger than '
                       f'{self.batch_size}, and `channel` equals to '
                       f'{self.image_channels}!\n'
                       f'But {images.shape} received!')

    xs = [self.transform(Image.fromarray(img)).unsqueeze(0) for img in images]
    xs = torch.cat(xs, dim=0).to(self.run_device)

    logits, features = self.net(xs)
    category_scores = self.get_value(F.softmax(logits, dim=1))
    features = self.get_value(features).squeeze(axis=(2, 3))
    attribute_scores = features.dot(self.attribute_weight.T)

    assert (len(category_scores.shape) == 2 and
            category_scores.shape[1] == NUM_CATEGORIES)
    assert (len(attribute_scores.shape) == 2 and
            attribute_scores.shape[1] == NUM_ATTRIBUTES)

    results = {
        'category': category_scores,
        'attribute': attribute_scores,
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def predict(self, images, **kwargs):
    return self.batch_run(images, self._predict)
