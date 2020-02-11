# python 3.7
"""Synthesize a collection of images with specified model."""

import os.path
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from models.helper import build_generator
from predictors.helper import build_predictor
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(description='Synthesize images with GAN.')
  parser.add_argument('model_name', type=str,
                      help='Name of the model used for synthesis.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`${MODEL_NAME}_synthesis` will be used by default.')
  parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (default: None)')
  parser.add_argument('-N', '--num', type=int, default=0,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is valid. Otherwise '
                           'a positive number is required. (default: 0)')
  parser.add_argument('--latent_space_type', type=str, default='z',
                      choices=['z', 'w', 'wp'],
                      help='Latent space used for synthesis in StyleGAN and '
                           'StyleGAN2. If the latent codes are loaded from '
                           'given path, they should align with the space type. '
                           '(default: `z`)')
  parser.add_argument('--skip_image', action='store_true',
                      help='If specified, will skip generating images in '
                           'StyleGAN and StyleGAN2. '
                           '(default: DO generate images)')
  parser.add_argument('--generate_style', action='store_true',
                      help='If specified, will generate layer-wise style codes '
                           'in StyleGAN and StyleGAN2. '
                           '(default: do NOT generate styles)')
  parser.add_argument('--generate_prediction', action='store_true',
                      help='If specified, will predict semantics from '
                           'synthesized images. (default: False)')
  parser.add_argument('--predictor_name', type=str, default='scene',
                      help='Name of the predictor used for analysis. (default: '
                           'scene)')
  parser.add_argument('--save_raw_synthesis', action='store_true',
                      help='If specified, will save raw synthesis to the disk. '
                           '(default: False)')
  parser.add_argument('--generate_html', action='store_true',
                      help='If specified, will use html for visualization. '
                           '(default: False)')
  parser.add_argument('--html_row', type=int, default=0,
                      help='Number of rows of the visualization html page. If '
                           'set as `0`, will be assigned based on number of '
                           'samples. (default: 0)')
  parser.add_argument('--html_col', type=int, default=0,
                      help='Number of columns of the visualization html page. '
                           'If set as `0`, will be assigned based on number of '
                           'samples. (default: 0)')
  parser.add_argument('--viz_size', type=int, default=0,
                      help='Image size for visualization on html page. Active '
                           'ONLY when `generate_html` is set as `True`. '
                           '`0` means to use the original synthesis size. '
                           '(default: 0)')
  parser.add_argument('--html_name', type=str, default='viz.html',
                      help='Name of the html page for visualization. Active '
                           'ONLY when `generate_html` is set as `True`. '
                           'If not specified, path `${OUTPUT_DIR}/viz.html` '
                           'will be used by default.')
  parser.add_argument('--logfile_name', type=str, default='log.txt',
                      help='Name of the log file. If not specified, log '
                           'message will be saved to path '
                           '`${OUTPUT_DIR}/log.txt` by default.')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()

  work_dir = args.output_dir or f'{args.model_name}_synthesis'
  logger_name = f'{args.model_name}_synthesis_logger'
  logger = setup_logger(work_dir, args.logfile_name, logger_name)

  logger.info(f'Initializing generator.')
  model = build_generator(args.model_name, logger=logger)

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.latent_codes_path):
    logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
    latent_codes = np.load(args.latent_codes_path)
    latent_codes = model.preprocess(latent_codes=latent_codes,
                                    latent_space_type=args.latent_space_type)
  else:
    if args.num <= 0:
      raise ValueError(f'Argument `num` should be specified as a positive '
                       f'number since the latent code path '
                       f'`{args.latent_codes_path}` does not exist!')
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(num=args.num,
                                     latent_space_type=args.latent_space_type)
  total_num = latent_codes.shape[0]

  if args.generate_prediction:
    logger.info(f'Initializing predictor.')
    predictor = build_predictor(args.predictor_name)

  if args.generate_html:
    viz_size = None if args.viz_size == 0 else args.viz_size
    visualizer = HtmlPageVisualizer(num_rows=args.html_row,
                                    num_cols=args.html_col,
                                    grid_size=total_num,
                                    viz_size=viz_size)

  logger.info(f'Generating {total_num} samples.')
  results = defaultdict(list)
  predictions = defaultdict(list)
  pbar = tqdm(total=total_num, leave=False)
  for inputs in model.get_batch_inputs(latent_codes):
    outputs = model.easy_synthesize(latent_codes=inputs,
                                    latent_space_type=args.latent_space_type,
                                    generate_style=args.generate_style,
                                    generate_image=not args.skip_image)
    for key, val in outputs.items():
      if key == 'image':
        if args.generate_prediction:
          pred_outputs = predictor.easy_predict(val)
          for pred_key, pred_val in pred_outputs.items():
            predictions[pred_key].append(pred_val)
        for image in val:
          if args.save_raw_synthesis:
            save_image(os.path.join(work_dir, f'{pbar.n:06d}.jpg'), image)
          if args.generate_html:
            row_idx = pbar.n // visualizer.num_cols
            col_idx = pbar.n % visualizer.num_cols
            visualizer.set_cell(row_idx, col_idx, image=image)
          pbar.update(1)
      else:
        results[key].append(val)
    if 'image' not in outputs:
      pbar.update(inputs.shape[0])
  pbar.close()

  logger.info(f'Saving results.')
  if args.generate_html:
    visualizer.save(os.path.join(work_dir, args.html_name))
  for key, val in results.items():
    np.save(os.path.join(work_dir, f'{key}.npy'), np.concatenate(val, axis=0))
  if predictions:
    if args.predictor_name == 'scene':
      # Categories
      categories = np.concatenate(predictions['category'], axis=0)
      detailed_categories = {
          'score': categories,
          'name_to_idx': predictor.category_name_to_idx,
          'idx_to_name': predictor.category_idx_to_name,
      }
      np.save(os.path.join(work_dir, 'category.npy'), detailed_categories)
      # Attributes
      attributes = np.concatenate(predictions['attribute'], axis=0)
      detailed_attributes = {
          'score': attributes,
          'name_to_idx': predictor.attribute_name_to_idx,
          'idx_to_name': predictor.attribute_idx_to_name,
      }
      np.save(os.path.join(work_dir, 'attribute.npy'), detailed_attributes)
    else:
      for key, val in predictions.items():
        np.save(os.path.join(work_dir, f'{key}.npy'),
                np.concatenate(val, axis=0))


if __name__ == '__main__':
  main()
