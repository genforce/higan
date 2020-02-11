# python 3.7
"""Manipulates images from latent space."""

import os.path
import argparse
import numpy as np
from tqdm import tqdm

from models.helper import build_generator
from utils.logger import setup_logger
from utils.editor import parse_boundary_list
from utils.editor import get_layerwise_manipulation_strength
from utils.editor import manipulate
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import fuse_images
from utils.visualizer import VideoWriter
from utils.visualizer import save_image

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
      description='Manipulate images from latent space of GAN.',
      epilog=_ATTRIBUTE_LIST_DESCRIPTION,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('model_name', type=str,
                      help='Name of the model used for synthesis.')
  parser.add_argument('boundary_list_path', type=str,
                      help='A list of `(name, space_type): path` boundaries. '
                           'Please see the description below.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`${MODEL_NAME}_manipulation` will be used by '
                           'default.')
  parser.add_argument('-c', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (default: None)')
  parser.add_argument('--latent_space_type', type=str, default='z',
                      choices=['z', 'w', 'wp'],
                      help='Space type of the input latent codes. This field '
                           'will also be used for latent codes sampling if '
                           'needed. (default: `z`)')
  parser.add_argument('-N', '--num', type=int, default=0,
                      help='Number of samples to manipulate. This field will '
                           'be ignored if `latent_codes_path` is valid. '
                           'Otherwise, a positive number is required. '
                           '(default: 0)')
  parser.add_argument('-S', '--step', type=int, default=7,
                      help='Number of manipulation steps. (default: 7)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start distance for manipulation. (default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End distance for manipulation. (default: 3.0)')
  parser.add_argument('--layerwise_manipulation', action='store_true',
                      help='If specified, will use layer-wise manipulation. '
                           '(default: False)')
  parser.add_argument('--disable_manipulation_truncation', action='store_true',
                      help='If specified, will NOT eliminate the truncation '
                           'effect during manipulation. '
                           '(default: DO eliminate)')
  parser.add_argument('--manipulate_layers', type=str, default='',
                      help='Indices of the layers to perform manipulation. '
                           'Active ONLY when `layerwise_manipulation` is set '
                           'as `True`. If not specified, all layers will be '
                           'manipulated. More than one layers should be '
                           'separated by `,`. (default: None)')
  parser.add_argument('--save_raw_synthesis', action='store_true',
                      help='If specified, will save raw synthesis to the disk. '
                           '(default: False)')
  parser.add_argument('--generate_html', action='store_true',
                      help='If specified, will use html for visualization. '
                           '(default: False)')
  parser.add_argument('--html_name', type=str, default='viz.html',
                      help='Name of the html page for visualization. Active '
                           'ONLY when `generate_html` is set as `True`. '
                           'If not specified, path '
                           '`${OUTPUT_DIR}/${ATTR_NAME}_${SPACE_TYPE}_viz.html`'
                           ' will be used by default.')
  parser.add_argument('--generate_video', action='store_true',
                      help='If specified, will create a video for '
                           'visualization. (default: False)')
  parser.add_argument('--video_name', type=str, default='viz.avi',
                      help='Name of the video for visualization. Active ONLY '
                           'when `generate_video` is set as `True`. If not '
                           'specified, path '
                           '`${OUTPUT_DIR}/${ATTR_NAME}_${SPACE_TYPE}_viz.avi` '
                           'will be used by default.')
  parser.add_argument('--fps', type=int, default=24,
                      help='Frame per second of the created video. Active ONLY '
                           'when `generate_video` is set as `True`. (default: '
                           '24)')
  parser.add_argument('--row', type=int, default=0,
                      help='Number of rows used in the video. If not set, will '
                           'be assigned automatically. (default: 0)')
  parser.add_argument('--col', type=int, default=0,
                      help='Number of columns used in the video. If not set, '
                           'will be assigned automatically (default: 0)')
  parser.add_argument('--row_spacing', type=int, default=0,
                      help='Row spacing used in the video. (default: 0)')
  parser.add_argument('--col_spacing', type=int, default=0,
                      help='Column spacing used in the video. (default: 0)')
  parser.add_argument('--border_left', type=int, default=0,
                      help='Left border used in the video. (default: 0)')
  parser.add_argument('--border_right', type=int, default=0,
                      help='Right border used in the video. (default: 0)')
  parser.add_argument('--border_top', type=int, default=0,
                      help='Top border used in the video. (default: 0)')
  parser.add_argument('--border_bottom', type=int, default=0,
                      help='Bottom border used in the video. (default: 0)')
  parser.add_argument('--white_background', action='store_true',
                      help='Whether to use white background in the video. '
                           '(default: False)')
  parser.add_argument('--viz_size', type=int, default=0,
                      help='Image size for visualization on html page and on '
                           'created video. Active ONLY when `generate_html` or '
                           '`generate_video` is set as `True`. `0` means to '
                           'use the original synthesis size. (default: 0)')
  parser.add_argument('--logfile_name', type=str, default='log.txt',
                      help='Name of the log file. If not specified, log '
                           'message will be saved to path '
                           '`${OUTPUT_DIR}/log.txt` by default.')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()

  work_dir = args.output_dir or f'{args.model_name}_manipulation'
  logger_name = f'{args.model_name}_manipulation_logger'
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

  latent_codes = model.easy_synthesize(latent_codes=latent_codes,
                                       latent_space_type=args.latent_space_type,
                                       generate_style=False,
                                       generate_image=False)
  for key, val in latent_codes.items():
    np.save(os.path.join(work_dir, f'{key}.npy'), val)

  boundaries = parse_boundary_list(args.boundary_list_path)

  step = args.step + int(args.step % 2 == 0)  # Make sure it is an odd number.

  for boundary_info, boundary_path in boundaries.items():
    boundary_name, space_type = boundary_info
    logger.info(f'Boundary `{boundary_name}` from {space_type.upper()} space.')
    prefix = f'{boundary_name}_{space_type}'

    if args.generate_html:
      viz_size = None if args.viz_size == 0 else args.viz_size
      visualizer = HtmlPageVisualizer(
          num_rows=total_num, num_cols=step + 1, viz_size=viz_size)
      visualizer.set_headers(
          [''] +
          [f'Step {i - step // 2}' for i in range(step // 2)] +
          ['Origin'] +
          [f'Step {i + 1}' for i in range(step // 2)]
      )

    if args.generate_video:
      setup_images = model.easy_synthesize(
          latent_codes=latent_codes[args.latent_space_type],
          latent_space_type=args.latent_space_type)['image']
      fusion_kwargs = {
          'row': args.row,
          'col': args.col,
          'row_spacing': args.row_spacing,
          'col_spacing': args.col_spacing,
          'border_left': args.border_left,
          'border_right': args.border_right,
          'border_top': args.border_top,
          'border_bottom': args.border_bottom,
          'black_background': not args.white_background,
          'image_size': None if args.viz_size == 0 else args.viz_size,
      }
      setup_image = fuse_images(setup_images, **fusion_kwargs)
      video_writer = VideoWriter(
          os.path.join(work_dir, f'{prefix}_{args.video_name}'),
          frame_height=setup_image.shape[0],
          frame_width=setup_image.shape[1],
          fps=args.fps)

    logger.info(f'  Loading boundary.')
    try:
      boundary_file = np.load(boundary_path, allow_pickle=True).item()
      boundary = boundary_file['boundary']
      manipulate_layers = boundary_file['meta_data']['manipulate_layers']
    except ValueError:
      boundary = np.load(boundary_path)
      manipulate_layers = args.manipulate_layers
    logger.info(f'  Manipulating on layers `{manipulate_layers}`.')

    np.save(os.path.join(work_dir, f'{prefix}_boundary.npy'), boundary)

    if args.layerwise_manipulation and space_type != 'z':
      layerwise_manipulation = True
      is_code_layerwise = True
      is_boundary_layerwise = (space_type == 'wp')
      if (not args.disable_manipulation_truncation) and space_type == 'w':
        strength = get_layerwise_manipulation_strength(
            model.num_layers, model.truncation_psi, model.truncation_layers)
      else:
        strength = 1.0
      space_type = 'wp'
    else:
      if args.layerwise_manipulation:
        logger.warning(f'  Skip layer-wise manipulation for boundary '
                       f'`{boundary_name}` from Z space. Traditional '
                       f'manipulation is used instead.')
      layerwise_manipulation = False
      is_code_layerwise = False
      is_boundary_layerwise = False
      strength = 1.0

    codes = manipulate(latent_codes=latent_codes[space_type],
                       boundary=boundary,
                       start_distance=args.start_distance,
                       end_distance=args.end_distance,
                       step=step,
                       layerwise_manipulation=layerwise_manipulation,
                       num_layers=model.num_layers,
                       manipulate_layers=manipulate_layers,
                       is_code_layerwise=is_code_layerwise,
                       is_boundary_layerwise=is_boundary_layerwise,
                       layerwise_manipulation_strength=strength)
    np.save(os.path.join(work_dir, f'{prefix}_manipulated_{space_type}.npy'),
            codes)

    logger.info(f'  Start manipulating.')
    for s in tqdm(range(step), leave=False):
      images = model.easy_synthesize(
          latent_codes=codes[:, s],
          latent_space_type=space_type)['image']
      if args.generate_video:
        video_writer.write(fuse_images(images, **fusion_kwargs))
      for n, image in enumerate(images):
        if args.save_raw_synthesis:
          save_image(os.path.join(work_dir, f'{prefix}_{n:05d}_{s:03d}.jpg'),
                     image)
        if args.generate_html:
          visualizer.set_cell(n, s + 1, image=image)
          if s == 0:
            visualizer.set_cell(n, 0, text=f'Sample {n:05d}')

    if args.generate_html:
      visualizer.save(os.path.join(work_dir, f'{prefix}_{args.html_name}'))

if __name__ == '__main__':
  main()
