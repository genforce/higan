# python 3.7
"""Contains the base class for semantic predictor."""

import os.path
import numpy as np

import torch

from . import predictor_settings

__all__ = ['BasePredictor']


class BasePredictor(object):
  """Base class for predictor used for analyzing semantics in images."""

  def __init__(self, predictor_name):
    """Initializes with specific settings.

    The predictor should be first registered in `predictor_settings.py` with
    proper settings. Among them, `some attributes are necessary, including:
`
    (1) weight_path: Path to the pre-trained weight.
    (2) resolution: Resolution of the input image required by the predictor.
                    `None` means not specified and the model will resize the
                    input image automatically before inference. (default: None)
    (3) image_channels: Number of channels of the input image required by the
                        predictor. (default: 3)
    (4) channel_order: Channel order of the input image required by the
                       predictor. (default: `RGB`)

    Args:
      predictor_name: Name with which the predictor is registered.

    Raises:
      AttributeError: If some necessary attributes are missing.
    """
    self.predictor_name = predictor_name

    # Parse settings.
    for key, val in predictor_settings.PREDICTOR_POOL[predictor_name].items():
      setattr(self, key, val)
    self.use_cuda = predictor_settings.USE_CUDA and torch.cuda.is_available()
    self.batch_size = predictor_settings.MAX_IMAGES_ON_DEVICE
    self.ram_size = predictor_settings.MAX_IMAGES_ON_RAM
    self.net = None
    self.run_device = 'cuda' if self.use_cuda else 'cpu'
    self.cpu_device = 'cpu'

    # Check necessary settings.
    self.check_attr('weight_path')
    assert os.path.isfile(self.weight_path)
    self.resolution = getattr(self, 'resolution', None)
    self.image_channels = getattr(self, 'image_channels', 3)
    assert self.image_channels in [1, 3]
    self.channel_order = getattr(self, 'channel_order', 'RGB').upper()
    assert self.channel_order in ['RGB', 'BGR']

    # Build graph and load pre-trained weights.
    self.build()
    self.load()

    # Change to inference mode and GPU mode if needed.
    assert self.net
    self.net.eval().to(self.run_device)

  def check_attr(self, attr_name):
    """Checks the existence of a particular attribute.

    Args:
      attr_name: Name of the attribute to check.

    Raises:
      AttributeError: If the target attribute is missing.
    """
    if not hasattr(self, attr_name):
      raise AttributeError(f'Field `{attr_name}` is missing for '
                           f'generator in model `{self.model_name}`!')

  def build(self):
    """Builds the graph."""
    raise NotImplementedError(f'Should be implemented in derived class!')

  def load(self):
    """Loads pre-trained weights."""
    raise NotImplementedError(f'Should be implemented in derived class!')

  def get_value(self, tensor):
    """Gets value of a `torch.Tensor`.

    Args:
      tensor: The input tensor to get value from.

    Returns:
      A `numpy.ndarray`.

    Raises:
      ValueError: If the tensor is with neither `torch.Tensor` type or
        `numpy.ndarray` type.
    """
    dtype = type(tensor)
    if isinstance(tensor, np.ndarray):
      return tensor
    if isinstance(tensor, torch.Tensor):
      return tensor.to(self.cpu_device).detach().numpy()
    raise ValueError(f'Unsupported input type `{dtype}`!')

  def get_batch_inputs(self, inputs, batch_size=None):
    """Gets inputs within mini-batch.

    This function yields at most `self.batch_size` inputs at a time.

    Args:
      inputs: Input data to form mini-batch.
      batch_size: Batch size. If not specified, `self.batch_size` will be used.
        (default: None)
    """
    total_num = inputs.shape[0]
    batch_size = batch_size or self.batch_size
    for i in range(0, total_num, batch_size):
      yield inputs[i:i + batch_size]

  def batch_run(self, inputs, run_fn):
    """Runs model with mini-batch.

    This function splits the inputs into mini-batches, run the model with each
    mini-batch, and then concatenate the outputs from all mini-batches together.

    NOTE: The output of `run_fn` can only be `numpy.ndarray` or a dictionary
    whose values are all `numpy.ndarray`.

    Args:
      inputs: The input samples to run with.
      run_fn: A callable function.

    Returns:
      Same type as the output of `run_fn`.

    Raises:
      ValueError: If the output type of `run_fn` is not supported.
    """
    if inputs.shape[0] > self.ram_size:
      self.logger.warning(f'Number of inputs on RAM is larger than '
                          f'{self.ram_size}. Please use '
                          f'`self.get_batch_inputs()` to split the inputs! '
                          f'Otherwise, it may encounter OOM problem!')

    results = {}
    temp_key = '__temp_key__'
    for batch_inputs in self.get_batch_inputs(inputs):
      batch_outputs = run_fn(batch_inputs)
      if isinstance(batch_outputs, dict):
        for key, val in batch_outputs.items():
          if not isinstance(val, np.ndarray):
            raise ValueError(f'Each item of the model output should be with '
                             f'type `numpy.ndarray`, but type `{type(val)}` is '
                             f'received for key `{key}`!')
          if key not in results:
            results[key] = [val]
          else:
            results[key].append(val)
      elif isinstance(batch_outputs, np.ndarray):
        if temp_key not in results:
          results[temp_key] = [batch_outputs]
        else:
          results[temp_key].append(batch_outputs)
      else:
        raise ValueError(f'The model output can only be with type '
                         f'`numpy.ndarray`, or a dictionary of '
                         f'`numpy.ndarray`, but type `{type(batch_outputs)}` '
                         f'is received!')

    for key, val in results.items():
      results[key] = np.concatenate(val, axis=0)
    return results if temp_key not in results else results[temp_key]

  def preprocess(self, images):
    """Preprocesses the input images if needed.

    This function assumes the input numpy array is with shape [batch_size,
    height, width, channel]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. Then, the function will check the shape
    of input images and adjust channel order.

    NOTE: The input images are always assumed to be with type `np.uint8`, range
    [0, 255], and channel order `RGB`.

    Args:
      images: The raw inputs with dtype `numpy.uint8`, range [0, 255], and
        channel order `RGB`.

    Returns:
      The preprocessed images with dtype `numpy.uint8`, range [0, 255], and
        channel order `self.channel_order`.

    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not
        with dtype `numpy.uint8` or not with shape [batch_size, height, width,
        channel].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')
    if images.dtype != np.uint8:
      raise ValueError(f'Images should be with dtype `numpy.uint8`!')

    if len(images.shape) != 4 or images.shape[3] != self.image_channels:
      raise ValueError(f'Input should be with shape [batch_size, height, '
                       f'width, channel], where channel equals to '
                       '{self.image_channels}!\n'
                       f'But {images.shape} is received!')
    if self.image_channels == 3 and self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    if self.resolution is not None:
      if isinstance(self.resolution, (list, tuple)):
        assert len(self.resolution) == 2
        height = self.resolution[0]
        width = self.resolution[1]
      else:
        assert isinstance(self.resolution, int)
        height = self.resolution
        width = self.resolution
      if images.shape[1] != height or images.shape[2] != width:
        raise ValueError(f'Input images should be with resolution [{height}, '
                         f'{width}], but {images.shape[1:3]} is received!')

    return images

  def predict(self, images, **kwargs):
    """Predict semantic scores from the input images.

    NOTE: The images are assumed to have already been preprocessed.

    Args:
      images: Input images to predict semantics on.

    Returns:
      A dictionary whose values are raw outputs from the predictor. Keys of
        the dictionary are usually the name of semantics.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_predict(self, images, **kwargs):
    """Wraps functions `preprocess()` and `predict()` together."""
    return self.predict(self.preprocess(images), **kwargs)
