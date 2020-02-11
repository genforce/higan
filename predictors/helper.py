# python 3.7
"""Helper function to build predictor."""

from .predictor_settings import PREDICTOR_POOL
from .scene_predictor import ScenePredictor

__all__ = ['build_predictor']


def build_predictor(predictor_name):
  """Builds predictor by predictor name."""
  if not predictor_name in PREDICTOR_POOL:
    raise ValueError(f'Model `{predictor_name}` is not registered in '
                     f'`PREDICTOR_POOL` in `predictor_settings.py`!')

  if predictor_name == 'scene':
    return ScenePredictor()
  raise NotImplementedError(f'Unsupported predictor `{predictor_name}`!')
