# python 3.7
"""Simple version for image manipulation."""

import os
import argparse
import numpy as np
from tqdm import tqdm

from models.helper import build_generator
from utils.logger import setup_logger
from utils.editor import get_layerwise_manipulation_strength
from utils.editor import manipulate
from utils.visualizer import HtmlPageVisualizer


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Simple image manipulation.')
  parser.add_argument('model_name', type=str,
                      help='Name of the model used for synthesis.')
  parser.add_argument('boundary_name', type=str,
                      help='Name of the boundary to manipulate.')
  parser.add_argument('-c', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (default: None)')
  parser.add_argument('--latent_space_type', type=str, default='w',
                      choices=['z', 'w', 'wp'],
                      help='Space type of the input latent codes. This field '
                           'will also be used for latent codes sampling if '
                           'needed. (default: `w`)')
  parser.add_argument('-N', '--num', type=int, default=10,
                      help='Number of samples to manipulate. This field will '
                           'be ignored if `latent_codes_path` is valid. '
                           'Otherwise, a positive number is required. '
                           '(default: 10)')
  parser.add_argument('-S', '--step', type=int, default=5,
                      help='Number of manipulation steps. (default: 5)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start distance for manipulation. (default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End distance for manipulation. (default: 3.0)')
  parser.add_argument('--manipulate_layers', type=str, default='6-11',
                      help='Indices of the layers to perform manipulation. '
                           'Active ONLY when `layerwise_manipulation` is set '
                           'as `True`. If not specified, all layers will be '
                           'manipulated. More than one layers should be '
                           'separated by `,`. (default: None)')
  args = parser.parse_args()

  work_dir = 'manipulation_results'
  os.makedirs(work_dir, exist_ok=True)
  prefix = f'{args.model_name}_{args.boundary_name}'
  logger = setup_logger(work_dir, '', 'logger')

  logger.info(f'Initializing generator.')
  model = build_generator(args.model_name, logger=logger)

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.latent_codes_path):
    logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
    latent_codes = np.load(args.latent_codes_path)
    latent_codes = model.preprocess(latent_codes=latent_codes,
                                    latent_space_type=args.latent_space_type)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(num=args.num,
                                     latent_space_type=args.latent_space_type)
  total_num = latent_codes.shape[0]

  latent_codes = model.easy_synthesize(latent_codes=latent_codes,
                                       latent_space_type=args.latent_space_type,
                                       generate_style=False,
                                       generate_image=False)
  for key, val in latent_codes.items():
    np.save(os.path.join(work_dir, f'{prefix}_{key}.npy'), val)

  logger.info(f'Loading boundary.')
  path = f'boundaries/{args.model_name}/{args.boundary_name}_boundary.npy'
  try:
    boundary_file = np.load(path, allow_pickle=True).item()
    boundary = boundary_file['boundary']
    manipulate_layers = boundary_file['meta_data']['manipulate_layers']
  except ValueError:
    boundary = np.load(path)
    manipulate_layers = args.manipulate_layers
  logger.info(f'  Manipulating on layers `{manipulate_layers}`.')

  np.save(os.path.join(work_dir, f'{prefix}_boundary.npy'), boundary)

  step = args.step + int(args.step % 2 == 0)  # Make sure it is an odd number.
  visualizer = HtmlPageVisualizer(num_rows=total_num, num_cols=step + 1)
  visualizer.set_headers(
      [''] +
      [f'Step {i - step // 2}' for i in range(step // 2)] +
      ['Origin'] +
      [f'Step {i + 1}' for i in range(step // 2)]
  )
  for n in range(total_num):
    visualizer.set_cell(n, 0, text=f'Sample {n:05d}')

  strength = get_layerwise_manipulation_strength(
      model.num_layers, model.truncation_psi, model.truncation_layers)
  codes = manipulate(latent_codes=latent_codes['wp'],
                     boundary=boundary,
                     start_distance=args.start_distance,
                     end_distance=args.end_distance,
                     step=step,
                     layerwise_manipulation=True,
                     num_layers=model.num_layers,
                     manipulate_layers=manipulate_layers,
                     is_code_layerwise=True,
                     is_boundary_layerwise=False,
                     layerwise_manipulation_strength=strength)
  np.save(os.path.join(work_dir, f'{prefix}_manipulated_wp.npy'), codes)

  for s in tqdm(range(step), leave=False):
    images = model.easy_synthesize(codes[:, s], latent_space_type='wp')['image']
    for n, image in enumerate(images):
      visualizer.set_cell(n, s + 1, image=image)
  visualizer.save(os.path.join(work_dir, f'{prefix}.html'))
