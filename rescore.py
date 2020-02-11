# python 3.7
"""Identifies the most relevant semantics with rescoring technique."""

import os.path
import argparse
import numpy as np
from tqdm import tqdm

from models.helper import build_generator
from utils.logger import setup_logger
from utils.editor import parse_boundary_list
from utils.editor import get_layerwise_manipulation_strength
from utils.editor import manipulate
from predictors.helper import build_predictor

_ATTRIBUTE_LIST_DESCRIPTION = '''
Attribute list desctipiton:

  Attribute list should be like:

    (age, z): $AGE_BOUNDARY_PATH
    (gender, w): $GENDER_BOUNDARY_PATH
    DISABLE(pose, wp): $POSE_BOUNDARY_PATH

  where the pose boundary from WP space will be ignored.
'''


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Identifies relevant semantics with rescoring technique.',
      epilog=_ATTRIBUTE_LIST_DESCRIPTION,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('model_name', type=str,
                      help='Name of the model used for synthesis.')
  parser.add_argument('boundary_list_path', type=str,
                      help='A list of `(name, space_type): path` boundaries. '
                           'Please see the description below.')
  parser.add_argument('--predictor_name', type=str, default='scene',
                      help='Name of the predictor used for analysis. (default: '
                           'scene)')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`${MODEL_NAME}_rescore` will be used by default.')
  parser.add_argument('-N', '--num', type=int, default=2000,
                      help='Number of samples used for identification. '
                           '(default: 2000)')
  parser.add_argument('--layerwise_rescoring', action='store_true',
                      help='If specified, will perform rescoring technique '
                           'layer-wise. (default: False)')
  parser.add_argument('--logfile_name', type=str, default='log.txt',
                      help='Name of the log file. If not specified, log '
                           'message will be saved to path '
                           '`${OUTPUT_DIR}/log.txt` by default.')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()

  work_dir = args.output_dir or f'{args.model_name}_rescore'
  logger_name = f'{args.model_name}_rescore_logger'
  logger = setup_logger(work_dir, args.logfile_name, logger_name)

  logger.info(f'Initializing generator.')
  model = build_generator(args.model_name, logger=logger)

  logger.info(f'Preparing latent codes.')
  if args.num <= 0:
    raise ValueError(f'Argument `num` should be specified as a positive '
                     f'number, but `{args.num}` received!')
  latent_codes = model.easy_sample(num=args.num, latent_space_type='z')
  latent_codes = model.easy_synthesize(latent_codes=latent_codes,
                                       latent_space_type='z',
                                       generate_style=False,
                                       generate_image=False)
  for key, val in latent_codes.items():
    np.save(os.path.join(work_dir, f'{key}.npy'), val)

  logger.info(f'Initializing predictor.')
  predictor = build_predictor(args.predictor_name)

  boundaries = parse_boundary_list(args.boundary_list_path)

  logger.info(f'========================================')
  logger.info(f'Rescoring.')
  score_changing = []
  for boundary_info, boundary_path in boundaries.items():
    logger.info(f'----------------------------------------')
    boundary_name, space_type = boundary_info
    logger.info(f'Boundary `{boundary_name}` from {space_type.upper()} space.')
    prefix = f'{boundary_name}_{space_type}'
    attr_idx = predictor.attribute_name_to_idx[boundary_name]

    try:
      boundary_file = np.load(boundary_path, allow_pickle=True).item()
      boundary = boundary_file['boundary']
    except ValueError:
      boundary = np.load(boundary_path)

    np.save(os.path.join(work_dir, f'{prefix}_boundary.npy'), boundary)

    if space_type == 'z':
      layerwise_manipulation = False
      is_code_layerwise = False
      is_boundary_layerwise = False
      num_layers = 0
      strength = 1.0
    else:
      layerwise_manipulation = True
      is_code_layerwise = True
      is_boundary_layerwise = (space_type == 'wp')
      num_layers = model.num_layers if args.layerwise_rescoring else 0
      if space_type == 'w':
        strength = get_layerwise_manipulation_strength(
            model.num_layers, model.truncation_psi, model.truncation_layers)
      else:
        strength = 1.0
      space_type = 'wp'

    codes = []
    codes.append(latent_codes[space_type][:, np.newaxis])
    for l in range(-1, num_layers):
      codes.append(manipulate(latent_codes[space_type],
                              boundary,
                              start_distance=2.0,
                              end_distance=2.0,
                              step=1,
                              layerwise_manipulation=layerwise_manipulation,
                              num_layers=model.num_layers,
                              manipulate_layers=None if l < 0 else l,
                              is_code_layerwise=is_code_layerwise,
                              is_boundary_layerwise=is_boundary_layerwise,
                              layerwise_manipulation_strength=strength))
    codes = np.concatenate(codes, axis=1)

    scores = []
    for i in tqdm(range(args.num), leave=False):
      images = model.easy_synthesize(latent_codes=codes[i],
                                     latent_space_type=space_type,
                                     generate_style=False,
                                     generate_image=True)['image']
      scores.append(predictor.easy_predict(images)['attribute'][:, attr_idx])
    scores = np.stack(scores, axis=0)
    np.save(os.path.join(work_dir, f'{prefix}_scores.npy'), scores)

    delta = scores[:, 1] - scores[:, 0]
    delta[delta < 0] = 0
    score_changing.append((boundary_name, np.mean(delta)))
    if num_layers:
      layerwise_score_changing = []
      for l in range(num_layers):
        delta = scores[:, l + 2] - scores[:, 0]
        delta[delta < 0] = 0
        layerwise_score_changing.append((f'Layer {l:02d}', np.mean(delta)))
      layerwise_score_changing.sort(key=lambda x: x[1], reverse=True)
      for layer_name, delta_score in layerwise_score_changing:
        logger.info(f'  {layer_name}: {delta_score:7.4f}')
  logger.info(f'----------------------------------------')
  logger.info(f'Most relevant semantics:')
  score_changing.sort(key=lambda x: x[1], reverse=True)
  for boundary_name, delta_score in score_changing:
    logger.info(f'  {boundary_name.ljust(15)}: {delta_score:7.4f}')


if __name__ == '__main__':
  main()
