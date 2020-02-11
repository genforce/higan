# python 3.7
"""Utility functions for semantic boundary search in latent space.

Currently, this script ONLY supports training linear boundary with SVM.
"""

import numpy as np
from sklearn import svm

from .logger import setup_logger

__all__ = ['train_boundary', 'project_boundary']


def convert_scores_to_binary_labels(scores,
                                    decision_value=None,
                                    separation_idx=None,
                                    is_score_ascending=False):
  """Converts given scores to binary labels.

  This function will return a binary vector, whose dimension is determined by
  the first dimension of the input `scores`. Basically, the labels are generated
  by checking the relationship between the `scores` and the `decision_value.`

  More concretely, all values that are larger than `decision_value` will be
  assigned label `positive (True)`, those smaller than or equal to
  `devision_value` will be assigned label `negative (False)`.

  This function also supports a simple implementation by assuming the input
  `scores` are already sorted. In this case, only an index is enough for label
  conversion, meaning that all samples in range [0, separation_idx) will be
  assigned to one class, and those in range [separation, total_num) will be
  assigned to another class. `is_score_ascending` is used to determine whether
  top half samples are `positive (True)` or `negative (False)`.

  Args:
    scores: Input scores as reference to label conversion. Should be with shape
      [num] or [num, 1].
    decision_value: A number to determine whether a sample belongs to positive
      class of negative class. This field will be ignored if `separation_idx` is
      specified. (default: None)
    separation_idx: Index used to separate the samples apart. This field can be
      used for a quick conversion when the scores are already sorted. (default:
      None)
    is_score_ascending: Whether the input scores are with ascending order. This
      field is only active when `separation_idx` is specified. The top half
      samples are assigned as `negative (False)` when this field is set as
      `True`, while assigned as `positive (False)` when this field is set as
      `False`. (default: False)

  Returns
    A binary vector with shape [num] as labels.

  Raises:
    ValueError: If both `decision_value` and `separation_idx` are missing.
  """
  num = scores.shape[0]
  labels = np.zeros(num, dtype=np.bool)

  if separation_idx is not None:
    separation_idx = np.clip(separation_idx, 0, num)
    if is_score_ascending:
      labels[separation_idx:] = True
    else:
      labels[:separation_idx] = True
  elif decision_value is not None:
    assert scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1)
    if scores.ndim == 1:
      labels[scores > decision_value] = True
    elif scores.ndim == 2:
      labels[scores[:, 0] > decision_value] = True
  else:
    raise ValueError('Both `decision_value` and `separation_idx` are missing!')

  return labels


def train_linear_boundary(data,
                          labels=None,
                          scores=None,
                          invalid_value=None,
                          chosen_num_or_ratio=10000,
                          split_ratio=0.7,
                          verbose_test=False,
                          logger=None):
  """Trains linear boundary with given data as a bi-classification problem.

  Given data with shape [num, dim], where `num` is the total number of samples
  and `dim` is the dimension of data space, this function returns a boundary
  that is able to separate the data into two classes based on the corresponding
  `labels` or `scores`.

  In the training process, partial data will be first selected from the entire
  dataset as the highly-convincing dataset. Then, this dataset is further
  splited to training and validation set for boundary search. After getting the
  bounary, this function will do evaluation on the validation set (required) as
  well as the remaining not-convincing set (optional). As for the convincing
  data selection, those samples with highest or lowest scores will be chosen.

  Args:
    data: Input data for training. Should be with shape [num, dim].
    labels: Binary labels corresponding to the input data. Should be with shape
      [num] or [num, 1]. (default: None)
    scores: Continuous labels corresponding to the input data. Should be with
      shape [num] or [num, 1]. This field will be ignored if `labels` is given.
      (default: None)
    invalid_value: Samples whose labels or scores are equal to this field will
      be filtered out. (default: None)
    chosen_num_or_ratio: Number of samples that will be chosen as positive (or
      negative) samples. If this field lies in range (0, 1], the number will be
      computed based on the total number of samples. Active only when `scores`
      is used. (default: 10000)
    split_ratio: Ratio to split training and validation sets. (default: 0.7)
    verbose_test: Whether to do evaluation on unconvincing dataset. Setting this
      field as `True` is time-consuming, but gives more detailed analysis.
      (default: False)
    logger: Logger for recording log messages. If set as `None`, a default
      logger, which prints messages from all levels to screen, will be created.
      (default: None)

  Returns:
    A decision boundary with shape (1, dim), dtype `numpy.float32` and
      normalized to unit norm.

  Raises:
    ValueError: If the input `data`, `labels`, or `scores` are with invalid
      shape. Or if both `labels` and `scores` are missing. Or if the input
      `chosen_num_or_ratio` is non-positive.
  """
  if not logger:
    logger = setup_logger(logfile_name=None, logger_name='boundary_logger')

  if data.ndim != 2:
    raise ValueError(f'Input `data` should be with shape [num, dim], but '
                     f'{data.shape} is received!')
  num, dim = data.shape

  if labels is not None:
    if labels.shape[0] != num or labels.ndim > 2 or labels.size != num:
      raise ValueError(f'Input `labels` should be with shape [num] or '
                       f'[num, 1], but {labels.shape} is received!')
    if labels.ndim == 2:
      labels = labels[:, 0]
    if invalid_value is not None:
      data = data[labels != invalid_value]
      labels = labels[labels != invalid_value]
    num = labels.shape[0]
    labels = labels.astype(np.bool)
    positive_idx = np.where(labels)[0]
    negative_idx = np.where(~labels)[0]

  elif scores is not None:
    if scores.shape[0] != num or scores.ndim > 2 or scores.size != num:
      raise ValueError(f'Input `scores` should be with shape [num] or '
                       f'[num, 1], but {scores.shape} is received!')
    if scores.ndim == 2:
      scores = scores[:, 0]
    if invalid_value is not None:
      data = data[scores != invalid_value]
      scores = scores[scores != invalid_value]
    _sorted_idx = np.argsort(scores)[::-1]  # Descending order.
    data = data[_sorted_idx]
    scores = scores[_sorted_idx]
    num = scores.shape[0]
    if chosen_num_or_ratio <= 0:
      raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                       f'but {chosen_num_or_ratio} received!')
    if 0 < chosen_num_or_ratio <= 1:
      chosen_num = int(num * chosen_num_or_ratio)
    else:
      chosen_num = int(chosen_num_or_ratio)
    chosen_num = min(chosen_num, num // 2)
    positive_idx = np.arange(0, chosen_num)
    negative_idx = np.arange(num - chosen_num, num)
    remaining_idx = np.arange(chosen_num, num - chosen_num)

  else:
    raise ValueError('Both `labels` and `scores` are missing!')

  if positive_idx.size == 0 or negative_idx.size == 0:
    raise ValueError(f'Input data only contains one class!')

  logger.info(f'Spliting training and validation sets:')
  logger.info(f'  Num of data after filtering: {num}.')
  positive_num = positive_idx.size
  negative_num = negative_idx.size
  remaining_num = num - positive_num - negative_num
  positive_train_num = int(positive_num * split_ratio)
  negative_train_num = int(negative_num * split_ratio)
  train_num = positive_train_num + negative_train_num
  positive_val_num = positive_num - positive_train_num
  negative_val_num = negative_num - negative_train_num
  val_num = positive_val_num + negative_val_num

  np.random.shuffle(positive_idx)
  np.random.shuffle(negative_idx)
  positive_train_idx = positive_idx[:positive_train_num]
  negative_train_idx = negative_idx[:negative_train_num]
  positive_val_idx = positive_idx[positive_train_num:]
  negative_val_idx = negative_idx[negative_train_num:]
  train_idx = np.concatenate([positive_train_idx, negative_train_idx])
  val_idx = np.concatenate([positive_val_idx, negative_val_idx])

  train_data = data[train_idx]
  val_data = data[val_idx]
  if labels is not None:
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
  elif scores is not None:
    train_scores = scores[train_idx]
    val_scores = scores[val_idx]
    train_labels = convert_scores_to_binary_labels(
        train_scores,
        separation_idx=positive_train_num,
        is_score_ascending=False)
    val_labels = convert_scores_to_binary_labels(
        val_scores,
        separation_idx=positive_val_num,
        is_score_ascending=False)
  logger.info(f'  Training set ({train_num}): '
              f'{positive_train_num} positive, '
              f'{negative_train_num} negative.')
  logger.info(f'  Validation set ({val_num}): '
              f'{positive_val_num} positive, '
              f'{negative_val_num} negative.')

  if remaining_num:
    assert labels is None
    remaining_data = data[remaining_idx]
    remaining_scores = scores[remaining_idx]
    remaining_labels = convert_scores_to_binary_labels(
        remaining_scores,
        decision_value=(remaining_scores[0] + remaining_scores[-1]) / 2)
    positive_remaining_num = np.sum(remaining_labels)
    negetive_remaining_num = np.sum(~remaining_labels)
    assert remaining_num == positive_remaining_num + negetive_remaining_num
    logger.info(f'  Remaining set ({remaining_num}): '
                f'{positive_remaining_num} positive, '
                f'{negetive_remaining_num} negative.')

  logger.info(f'Training boundary.')
  clf = svm.SVC(kernel='linear')
  classifier = clf.fit(train_data, train_labels)
  logger.info(f'Finish training.')

  if val_num > 0:
    val_prediction = classifier.predict(val_data)
    positive_correct_num = np.sum(
        val_labels[val_labels] == val_prediction[val_labels])
    negative_correct_num = np.sum(
        val_labels[~val_labels] == val_prediction[~val_labels])
    correct_num = positive_correct_num + negative_correct_num
    logger.info(f'Accuracy for validation set:')
    logger.info(f'  Positive: {positive_correct_num} / {positive_val_num}'
                f' = {positive_correct_num / positive_val_num:.6f}')
    logger.info(f'  Negative: {negative_correct_num} / {negative_val_num}'
                f' = {negative_correct_num / negative_val_num:.6f}')
    logger.info(f'  All: {correct_num} / {val_num}'
                f' = {correct_num / val_num:.6f}')

  if remaining_num > 0 and verbose_test:
    remaining_prediction = classifier.predict(remaining_data)
    positive_correct_num = np.sum(remaining_labels[remaining_labels] ==
                                  remaining_prediction[remaining_labels])
    negative_correct_num = np.sum(remaining_labels[~remaining_labels] ==
                                  remaining_prediction[~remaining_labels])
    correct_num = positive_correct_num + negative_correct_num
    logger.info(f'Accuracy for remaining set:')
    logger.info(f'  Positive: {positive_correct_num} / {positive_remaining_num}'
                f' = {positive_correct_num / positive_remaining_num:.6f}')
    logger.info(f'  Negative: {negative_correct_num} / {negetive_remaining_num}'
                f' = {negative_correct_num / negetive_remaining_num:.6f}')
    logger.info(f'  All: {correct_num} / {remaining_num}'
                f' = {correct_num / remaining_num:.6f}')

  direction = classifier.coef_.reshape(1, dim).astype(np.float32)
  return direction / np.linalg.norm(direction)


def train_boundary(data,
                   labels=None,
                   scores=None,
                   boundary_type='linear',
                   logger=None,
                   **kwargs):
  """Trains boundary with given data as a classification problem.

  Args:
    data: Input data for training. Should be with shape [num, dim].
    labels: Binary labels corresponding to the input data. Should be with shape
      [num] or [num, 1]. (default: None)
    scores: Continuous labels corresponding to the input data. Should be with
      shape [num] or [num, 1]. (default: None)
    boundary_type: Type of the boundary which is desired as output. (default:
      `linear`)
    logger: Logger for recording log messages. If set as `None`, a default
      logger, which prints messages from all levels to screen, will be created.
      (default: None)

  Returns:
    A boundary trained with given data.
  """
  if boundary_type == 'linear':
    return train_linear_boundary(data=data,
                                 scores=scores,
                                 labels=labels,
                                 logger=logger,
                                 **kwargs)
  raise NotImplementedError(f'Unsupported boundary type `{boundary_type}`!')


def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.

  The function is used for conditional manipulation, where the projected vector
  will be subscribed from the normal direction of the original boundary. Here,
  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].

  NOTE: For now, at most two condition boundaries are supported.

  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.

  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.

  Raises:
    NotImplementedError: If there are more than two condition boundaries.
  """
  if len(args) > 2:
    raise NotImplementedError(f'This function supports projecting with at most '
                              f'two conditions.')
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal

  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)

  if len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)

  raise NotImplementedError
