# python 3.7
"""Trains boundary with given data and attribute scores.

Basically, given data with shape [num, dim], where `num` stands for total number
of samples and `dim` represents the space dimension for boundary search, and
scores (or say, continuous labels) with shape [num], this script trains a
boundary, with shape [1, dim], to separate the data.

Layer-wise boundary training is supported. More concretely, when given data is
with shape [num, ..., dim], this script will first reshape the data to shape
[num, num_layers, dim], and train an independent boundary on each `layer`.
The well-trained boundaries will be concatenated together and then reshaped to
shape [1, ..., dim].

NOTE: Currently, this script ONLY supports training linear boundary with SVM.
"""

import os.path
import argparse
import numpy as np

from utils.logger import setup_logger
from utils.boundary_searcher import train_boundary


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train boundary with given data and labels.')
  parser.add_argument('data_path', type=str,
                      help='Path to the data for training.')
  parser.add_argument('scores_path', type=str,
                      help='Path to the scores for training.')
  parser.add_argument('--score_name', type=str, default='',
                      help='Name of the score. If specified, file '
                           '`scores_path` is expected to be a dictionary, and '
                           'key `score_name` will be used as the scores for '
                           'training. (default: None)')
  parser.add_argument('-o', '--output_dir', type=str, default='results',
                      help='Directory to save the results. '
                           '(default: `results`)')
  parser.add_argument('-T', '--boundary_type', type=str, default='linear',
                      choices=['linear'],
                      help='Type of the boundary. (default: `linear`)')
  parser.add_argument('-N', '--chosen_num_or_ratio', type=float, default=2000,
                      help='Number of highly-convincing samples that will be '
                           'used for training. (default: 2000)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
  parser.add_argument('-v', '--invalid_value', type=float, default=None,
                      help='Samples whose score is equal to this value will be '
                           'ignored. (default: None)')
  parser.add_argument('-V', '--verbose_test', action='store_true',
                      help='Whether to do evaluation on the entire dataset '
                           'after getting the well_trained boundary. It may '
                           'be time-consuming. (default: False)')
  parser.add_argument('--save_name', type=str, default='boundary.npy',
                      help='Name of the well-trained boundary to be saved. '
                           'It will be saved to path '
                           '`${OUTPUT_DIR}/${SAVE_NAME}`.')
  parser.add_argument('--logfile_name', type=str, default='',
                      help='Name of the log file. If not specified, log '
                           'message will be saved to path '
                           '`${OUTPUT_DIR}/${SAVE_NAME}.log` '
                           'by default.')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()

  work_dir = args.output_dir
  if args.save_name[-4:] != '.npy':
    save_name = args.save_name + '.npy'
  else:
    save_name = args.save_name
  if args.score_name:
    save_name = args.score_name + '_' + save_name
  logfile_name = args.logfile_name or save_name[:-4] + '.log'
  logger_name = f'boundary_training_logger'
  logger = setup_logger(work_dir, logfile_name, logger_name)

  logger.info(f'Loading data from `{args.data_path}`.')
  if not os.path.isfile(args.data_path):
    raise ValueError(f'Data `{args.data_path}` does not exist!')
  data = np.load(args.data_path)

  logger.info(f'Loading scores from `{args.scores_path}`.')
  if not os.path.isfile(args.scores_path):
    raise ValueError(f'Scores `{args.scores_path}` does not exist!')
  scores = np.load(args.scores_path, allow_pickle=True)[()]
  if args.score_name:
    assert isinstance(scores, dict)
    if args.score_name in scores:
      scores = scores[args.score_name]
    else:
      score_idx = scores['name_to_idx'][args.score_name]
      scores = scores['score'][:, score_idx]

  if data.ndim < 2:
    raise ValueError(f'Data should be with shape [num, ..., dim], where `num` '
                     f'is the total number of smaples and `dim` is the space '
                     f'dimension for boundary search.\n'
                     f'But {data.ndim} is received!')
  data_shape = data.shape

  data = data.reshape(data_shape[0], -1, data_shape[-1])
  boundaries = []
  for layer_idx in range(data.shape[1]):
    logger.info(f'==== Layer {layer_idx:02d} ====')
    boundary = train_boundary(data=data[:, layer_idx],
                              scores=scores,
                              boundary_type=args.boundary_type,
                              invalid_value=args.invalid_value,
                              chosen_num_or_ratio=args.chosen_num_or_ratio,
                              split_ratio=args.split_ratio,
                              verbose_test=args.verbose_test,
                              logger=logger)
    boundaries.append(boundary)
  boundaries = np.stack(boundaries, axis=1)

  boundaries = boundaries.reshape(1, *data_shape[1:])
  np.save(os.path.join(work_dir, save_name), boundaries)


if __name__ == '__main__':
  main()
