# python 3.7
"""Unit tests.

This script can also be used as a sample of code usage.

TODO: StyleGAN and StyleGAN2 models scannot be converted simultaneously, since
the scripts in `stylegan_tf_official` and `stylegan2_tf_official` conflict to
each other.
"""

import os
import argparse
import numpy as np

from models.model_settings import USE_CUDA, MODEL_POOL
from models.helper import build_generator
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer, get_grid_shape
from utils.editor import parse_indices, interpolate, mix_style, manipulate
from utils.editor import get_layerwise_manipulation_strength
from utils.boundary_searcher import train_boundary, project_boundary

TEST_BATCH_SIZE = 1  # Small batch size in case of converting tensorflow weight.
RESULT_DIR = 'results'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      'Test modules defined in folder `models` and `units`.')
  parser.add_argument('--test_num', type=int, default=10,
                      help='Number of test samples. (default: 10)')
  parser.add_argument('--verbose', action='store_true',
                      help='Wether to test all availabel models. (default: '
                           'False)')
  parser.add_argument('--pggan', action='store_true',
                      help='Whether to test on PGGAN models. (default: False)')
  parser.add_argument('--stylegan', action='store_true',
                      help='Whether to test on StyleGAN models. (default: '
                           'False)')
  parser.add_argument('--stylegan2', action='store_true',
                      help='Whether to test on StyleGAN2 models. (default: '
                           'False)')
  parser.add_argument('--editor', action='store_true',
                      help='Whether to test the editing functions. (default: '
                           'False)')
  parser.add_argument('--boundary', action='store_true',
                      help='Whether to test the boundary related functions. '
                           '(default: False)')
  parser.add_argument('--all', action='store_true',
                      help='Whether to execute all test. (default: False)')
  args = parser.parse_args()

  if not USE_CUDA:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  TEST_FLAG = False

  ###########################
  #### Model Test Starts ####
  ###########################
  # PGGAN Generator.
  if args.pggan or args.all:
    print('==== PGGAN Generator Test ====')
    if args.verbose:
      model_list = []
      for model_name, model_setting in MODEL_POOL.items():
        if model_setting['gan_type'] == 'pggan':
          model_list.append(model_name)
    else:
      model_list = ['pggan_celebahq', 'pggan_bedroom']
    for model_name in model_list:
      logger = setup_logger(work_dir=RESULT_DIR,
                            logfile_name=f'{model_name}_generator_test.log',
                            logger_name=f'{model_name}_generator_logger')
      G = build_generator(model_name, logger=logger)
      G.batch_size = TEST_BATCH_SIZE
      z = G.easy_sample(args.test_num)
      x = G.easy_synthesize(z)['image']
      visualizer = HtmlPageVisualizer(grid_size=args.test_num)
      for i in range(visualizer.num_rows):
        for j in range(visualizer.num_cols):
          visualizer.set_cell(i, j, image=x[i * visualizer.num_cols + j])
      visualizer.save(f'{RESULT_DIR}/{model_name}_generator_test.html')
      del G
    print('Pass!')
    TEST_FLAG = True

  # StyleGAN Generator.
  if args.stylegan or args.all:
    print('==== StyleGAN Generator Test ====')
    if args.verbose:
      model_list = []
      for model_name, model_setting in MODEL_POOL.items():
        if model_setting['gan_type'] == 'stylegan':
          model_list.append(model_name)
    else:
      model_list = ['stylegan_ffhq', 'stylegan_car', 'stylegan_bedroom']
    for model_name in model_list:
      logger = setup_logger(work_dir=RESULT_DIR,
                            logfile_name=f'{model_name}_generator_test.log',
                            logger_name=f'{model_name}_generator_logger')
      G = build_generator(model_name, logger=logger)
      G.batch_size = TEST_BATCH_SIZE
      z = G.easy_sample(args.test_num)
      x = G.easy_synthesize(z)['image']
      visualizer = HtmlPageVisualizer(grid_size=args.test_num)
      for i in range(visualizer.num_rows):
        for j in range(visualizer.num_cols):
          visualizer.set_cell(i, j, image=x[i * visualizer.num_cols + j])
      visualizer.save(f'{RESULT_DIR}/{model_name}_generator_test.html')
      del G
    print('Pass!')
    TEST_FLAG = True

  # StyleGAN2 Generator.
  if args.stylegan2 or args.all:
    print('==== StyleGAN2 Generator Test ====')
    if args.verbose:
      model_list = []
      for model_name, model_setting in MODEL_POOL.items():
        if model_setting['gan_type'] == 'stylegan2':
          model_list.append(model_name)
    else:
      model_list = ['stylegan2_ffhq', 'stylegan2_car', 'stylegan2_church']
    for model_name in model_list:
      logger = setup_logger(work_dir=RESULT_DIR,
                            logfile_name=f'{model_name}_generator_test.log',
                            logger_name=f'{model_name}_generator_logger')
      G = build_generator(model_name, logger=logger)
      G.batch_size = TEST_BATCH_SIZE
      z = G.easy_sample(args.test_num)
      x = G.easy_synthesize(z)['image']
      visualizer = HtmlPageVisualizer(grid_size=args.test_num)
      for i in range(visualizer.num_rows):
        for j in range(visualizer.num_cols):
          visualizer.set_cell(i, j, image=x[i * visualizer.num_cols + j])
      visualizer.save(f'{RESULT_DIR}/{model_name}_generator_test.html')
      del G
    print('Pass!')
    TEST_FLAG = True
  #########################
  #### Model Test Ends ####
  #########################

  ############################
  #### Editor Test Starts ####
  ############################
  if args.editor or args.all:
    print('==== Grip Shape Test ====')
    assert get_grid_shape(0) == (0, 0)
    assert get_grid_shape(1) == (1, 1)
    assert get_grid_shape(10) == (2, 5)
    assert get_grid_shape(100) == (10, 10)
    assert get_grid_shape(17) == (1, 17)
    assert get_grid_shape(15) == (3, 5)
    assert get_grid_shape(24) == (4, 6)
    assert get_grid_shape(50) == (5, 10)
    assert get_grid_shape(512) == (16, 32)
    assert get_grid_shape(36) == (6, 6)
    assert get_grid_shape(36, row=12) == (12, 3)
    assert get_grid_shape(36, col=12) == (3, 12)
    assert get_grid_shape(36, row=12, col=12) == (6, 6)
    assert get_grid_shape(0, is_portrait=True) == (0, 0)
    assert get_grid_shape(1, is_portrait=True) == (1, 1)
    assert get_grid_shape(10, is_portrait=True) == (5, 2)
    assert get_grid_shape(100, is_portrait=True) == (10, 10)
    assert get_grid_shape(17, is_portrait=True) == (17, 1)
    assert get_grid_shape(15, is_portrait=True) == (5, 3)
    assert get_grid_shape(24, is_portrait=True) == (6, 4)
    assert get_grid_shape(50, is_portrait=True) == (10, 5)
    assert get_grid_shape(512, is_portrait=True) == (32, 16)
    assert get_grid_shape(36, row=12, is_portrait=True) == (12, 3)
    assert get_grid_shape(36, col=12, is_portrait=True) == (3, 12)
    assert get_grid_shape(36, row=12, col=12, is_portrait=True) == (6, 6)
    print('Pass!')

    print('==== Index Parser Test ====')
    assert parse_indices(None) == []
    assert parse_indices('') == []
    assert parse_indices([]) == []
    assert parse_indices(0) == [0]
    assert parse_indices('1,2,3') == [1, 2, 3]
    assert parse_indices('1, 2, 3') == [1, 2, 3]
    assert parse_indices('1, 2, 3, 5-7') == [1, 2, 3, 5, 6, 7]
    assert (parse_indices('1, 2, 3, 5-7, 10, 12, 15-16') ==
            [1, 2, 3, 5, 6, 7, 10, 12, 15, 16])
    assert (parse_indices('1, 5-7, 2, 3, 10, 12, 15-16, 20') ==
            [1, 2, 3, 5, 6, 7, 10, 12, 15, 16, 20])
    print('Pass!')

    num_layers = 18
    dim = 512

    print('==== Interpolation Test (single latent code)====')
    step = 5
    num = 16
    a = np.random.randint(0, high=10000, size=(num, dim))
    b = np.random.randint(0, high=10000, size=(num, dim))
    res = interpolate(a, b, step=step)
    assert res.shape == (num, step, dim)
    assert np.all(res[:, 0] == a)
    assert np.all(res[:, -1] == b)
    error = 0
    diff = (b - a) / (step - 1)
    for s in range(1, step):
      error += np.average(np.abs((res[:, s] - res[:, s - 1] - diff)))
    print('Error:', error)

    print('==== Interpolation Test (layer-wise latent code)====')
    step = 5
    num = 16
    a = np.random.randint(0, high=10000, size=(num, num_layers, dim))
    b = np.random.randint(0, high=10000, size=(num, num_layers, dim))
    res = interpolate(a, b, step=step)
    assert res.shape == (num, step, num_layers, dim)
    assert np.all(res[:, 0] == a)
    assert np.all(res[:, -1] == b)
    error = 0
    diff = (b - a) / (step - 1)
    for s in range(1, step):
      error += np.average(np.abs((res[:, s] - res[:, s - 1] - diff)))
    print('Error:', error)

    print('==== Style Mixing Test (single latent code) ====')
    s_num = 16
    c_num = 32
    indices = parse_indices('0-1', min_val=0, max_val=num_layers - 1)
    s = np.random.randint(0, high=10000, size=(s_num, dim))
    c = np.random.randint(0, high=10000, size=(c_num, dim))
    res = mix_style(style_codes=s,
                    content_codes=c,
                    num_layers=num_layers,
                    mix_layers=indices,
                    is_style_layerwise=False,
                    is_content_layerwise=False)
    assert res.shape == (s_num, c_num, num_layers, dim)
    error = 0
    for i in range(s_num):
      for j in range(c_num):
        for k in range(num_layers):
          if k in indices:
            error += np.average(np.abs((res[i, j, k] - s[i])))
          else:
            error += np.average(np.abs((res[i, j, k] - c[j])))
    print('Error:', error)

    print('==== Style Mixing Test (layer-wise latent code) ====')
    s_num = 32
    c_num = 16
    indices = parse_indices('3, 6, 9, 12', min_val=0, max_val=num_layers - 1)
    s = np.random.randint(0, high=10000, size=(s_num, num_layers, dim))
    c = np.random.randint(0, high=10000, size=(c_num, num_layers, dim))
    res = mix_style(style_codes=s,
                    content_codes=c,
                    num_layers=num_layers,
                    mix_layers=indices)
    assert res.shape == (s_num, c_num, num_layers, dim)
    error = 0
    for i in range(s_num):
      for j in range(c_num):
        for k in range(num_layers):
          if k in indices:
            error += np.average(np.abs((res[i, j, k] - s[i, k])))
          else:
            error += np.average(np.abs((res[i, j, k] - c[j, k])))
    print('Error:', error)

    print('==== Single Manipulation Test (single latent code) ====')
    num = 64
    start_distance = -10
    end_distance = -start_distance
    step = 21
    strength = 0.7
    x = np.random.randint(0, high=10000, size=(num, dim))
    b = np.random.randint(0, high=10000, size=(1, dim))
    res = manipulate(latent_codes=x,
                     boundary=b,
                     start_distance=start_distance,
                     end_distance=end_distance,
                     step=step,
                     layerwise_manipulation=False,
                     num_layers=1,
                     manipulate_layers=None,
                     is_code_layerwise=False,
                     is_boundary_layerwise=False,
                     layerwise_manipulation_strength=strength)
    assert res.shape == (num, step, dim)
    assert np.all(res[:, step // 2] == x)
    diff = (end_distance - start_distance) / (step - 1) * b[0]
    error = 0
    for i in range(num):
      for j in range(step):
        error += np.average(np.abs(res[i, j] - res[i, 0] - diff * j))
    print('Error:', error)

    print('==== Layer-wise Manipulation Test (single latent code, '
          'single boundary) ====')
    num = 64
    start_distance = -10
    end_distance = -start_distance
    step = 21
    truncation_psi = 1.0
    truncation_layers = 10
    strength = get_layerwise_manipulation_strength(
        num_layers, truncation_psi, truncation_layers)
    indices = parse_indices('0-8, 10-12', min_val=0, max_val=num_layers - 1)
    x = np.random.randint(0, high=10000, size=(num, dim))
    b = np.random.randint(0, high=10000, size=(1, dim))
    res = manipulate(latent_codes=x,
                     boundary=b,
                     start_distance=start_distance,
                     end_distance=end_distance,
                     step=step,
                     layerwise_manipulation=True,
                     num_layers=num_layers,
                     manipulate_layers=indices,
                     is_code_layerwise=False,
                     is_boundary_layerwise=False,
                     layerwise_manipulation_strength=strength)
    assert res.shape == (num, step, num_layers, dim)
    assert np.all(res[:, step // 2] == np.tile(x[:, np.newaxis],
                                               (1, num_layers, 1)))
    diff = (end_distance - start_distance) / (step - 1) * b[0]
    error = 0
    for i in range(num):
      for j in range(step):
        for k in range(num_layers):
          if k in indices:
            _diff = diff * (truncation_psi if k < truncation_layers else 1.0)
            error += np.average(np.abs(res[i, j, k] - res[i, 0, k] - _diff * j))
          else:
            error += np.average(np.abs(res[i, j, k] - x[i]))
    print('Error:', error)

    print('==== Layer-wise Manipulation Test (layer-wise latent code, '
          'single boundary) ====')
    num = 64
    start_distance = -10
    end_distance = -start_distance
    step = 21
    truncation_psi = 0.7
    truncation_layers = 0
    strength = get_layerwise_manipulation_strength(
        num_layers, truncation_psi, truncation_layers)
    indices = parse_indices('1, 4, 7, 9, 17', min_val=0, max_val=num_layers - 1)
    x = np.random.randint(0, high=10000, size=(num, num_layers, dim))
    b = np.random.randint(0, high=10000, size=(1, dim))
    res = manipulate(latent_codes=x,
                     boundary=b,
                     start_distance=start_distance,
                     end_distance=end_distance,
                     step=step,
                     layerwise_manipulation=True,
                     num_layers=num_layers,
                     manipulate_layers=indices,
                     is_code_layerwise=True,
                     is_boundary_layerwise=False,
                     layerwise_manipulation_strength=strength)
    assert res.shape == (num, step, num_layers, dim)
    assert np.all(res[:, step // 2] == x)
    diff = (end_distance - start_distance) / (step - 1) * b[0]
    error = 0
    for i in range(num):
      for j in range(step):
        for k in range(num_layers):
          if k in indices:
            _diff = diff * (truncation_psi if k < truncation_layers else 1.0)
            error += np.average(np.abs(res[i, j, k] - res[i, 0, k] - _diff * j))
          else:
            error += np.average(np.abs(res[i, j, k] - x[i, k]))
    print('Error:', error)

    print('==== Layer-wise Manipulation Test (single latent code, '
          'layer-wise boundary) ====')
    num = 64
    start_distance = -10
    end_distance = -start_distance
    step = 21
    truncation_psi = 0.5
    truncation_layers = 10
    strength = get_layerwise_manipulation_strength(
        num_layers, truncation_psi, truncation_layers)
    indices = parse_indices('0, 3, 17', min_val=0, max_val=num_layers - 1)
    x = np.random.randint(0, high=10000, size=(num, dim))
    b = np.random.randint(0, high=10000, size=(1, num_layers, dim))
    res = manipulate(latent_codes=x,
                     boundary=b,
                     start_distance=start_distance,
                     end_distance=end_distance,
                     step=step,
                     layerwise_manipulation=True,
                     num_layers=num_layers,
                     manipulate_layers=indices,
                     is_code_layerwise=False,
                     is_boundary_layerwise=True,
                     layerwise_manipulation_strength=strength)
    assert res.shape == (num, step, num_layers, dim)
    assert np.all(res[:, step // 2] == np.tile(x[:, np.newaxis],
                                               (1, num_layers, 1)))
    error = 0
    for i in range(num):
      for j in range(step):
        for k in range(num_layers):
          diff = (end_distance - start_distance) / (step - 1) * b[0, k]
          if k in indices:
            _diff = diff * (truncation_psi if k < truncation_layers else 1.0)
            error += np.average(np.abs(res[i, j, k] - res[i, 0, k] - _diff * j))
          else:
            error += np.average(np.abs(res[i, j, k] - x[i]))
    print('Error:', error)

    print('==== Layer-wise Manipulation Test (layer-wise latent code, '
          'layer-wise boundary) ====')
    num = 64
    start_distance = -10
    end_distance = -start_distance
    step = 21
    truncation_psi = 0.5
    truncation_layers = num_layers
    strength = get_layerwise_manipulation_strength(
        num_layers, truncation_psi, truncation_layers)
    indices = parse_indices('2, 5, 6-8, 10', min_val=0, max_val=num_layers - 1)
    x = np.random.randint(0, high=10000, size=(num, num_layers, dim))
    b = np.random.randint(0, high=10000, size=(1, num_layers, dim))
    res = manipulate(latent_codes=x,
                     boundary=b,
                     start_distance=start_distance,
                     end_distance=end_distance,
                     step=step,
                     layerwise_manipulation=True,
                     num_layers=num_layers,
                     manipulate_layers=indices,
                     is_code_layerwise=True,
                     is_boundary_layerwise=True,
                     layerwise_manipulation_strength=strength)
    assert res.shape == (num, step, num_layers, dim)
    assert np.all(res[:, step // 2] == x)
    error = 0
    for i in range(num):
      for j in range(step):
        for k in range(num_layers):
          diff = (end_distance - start_distance) / (step - 1) * b[0, k]
          if k in indices:
            _diff = diff * (truncation_psi if k < truncation_layers else 1.0)
            error += np.average(np.abs(res[i, j, k] - res[i, 0, k] - _diff * j))
          else:
            error += np.average(np.abs(res[i, j, k] - x[i, k]))
    print('Error:', error)

    TEST_FLAG = True
  #########################
  #### Editor Test End ####
  #########################


  #####################################
  #### Boundary Search Test Starts ####
  #####################################
  if args.boundary or args.all:
    dim = 512

    print('==== Boundary Projection Test (no condition) ====')
    a = np.random.randn(1, dim).astype(np.float32)
    a = a / np.linalg.norm(a)
    proj = project_boundary(a)
    print(f'Boundary Norm: {np.linalg.norm(proj)}')
    print(f'Error: {1 - np.sum(proj * a)}')

    print('==== Boundary Projection Test (single condition) ====')
    a = np.random.randn(1, dim).astype(np.float32)
    a = a / np.linalg.norm(a)
    b = np.random.randn(1, dim).astype(np.float32)
    b = b / np.linalg.norm(b)
    proj = project_boundary(a, b)
    print(f'Boundary Norm: {np.linalg.norm(proj)}')
    print(f'Error: {np.sum(proj * b)}')

    print('==== Boundary Projection Test (multiple conditions) ====')
    a = np.random.randn(1, dim).astype(np.float32)
    a = a / np.linalg.norm(a)
    b = np.random.randn(1, dim).astype(np.float32)
    b = b / np.linalg.norm(b)
    c = np.random.randn(1, dim).astype(np.float32)
    c = c / np.linalg.norm(c)
    proj = project_boundary(a, b, c)
    print(f'Boundary Norm: {np.linalg.norm(proj)}')
    print(f'Error: {np.sum(proj * b) + np.sum(proj * c)}')

    num = 1000
    chosen_ratio = 0.3
    data = np.random.randn(num, dim).astype(np.float32)
    logger = setup_logger(work_dir=RESULT_DIR,
                          logfile_name=f'boundary_search_test.log',
                          logger_name=f'boundary_search_logger')

    print('==== Boundary Search Test (using labels) ====')
    labels = (np.random.randn(num) > 0.5).astype(np.bool)
    boundary = train_boundary(data,
                              labels=labels,
                              verbose_test=True,
                              logger=logger)
    assert boundary.shape == (1, dim)
    print(f'Boundary norm: {np.linalg.norm(boundary)}')

    print('==== Boundary Search Test (using scores with chosen num) ====')
    scores = np.random.randn(num).astype(np.float32)
    boundary = train_boundary(data,
                              scores=scores,
                              chosen_num_or_ratio=chosen_ratio * num,
                              verbose_test=True,
                              logger=logger)
    assert boundary.shape == (1, dim)
    print(f'Boundary norm: {np.linalg.norm(boundary)}')

    print('==== Boundary Search Test (using scores with chosen ratio) ====')
    scores = np.random.randn(num).astype(np.float32)
    boundary = train_boundary(data,
                              scores=scores,
                              chosen_num_or_ratio=chosen_ratio,
                              verbose_test=True,
                              logger=logger)
    assert boundary.shape == (1, dim)
    print(f'Boundary norm: {np.linalg.norm(boundary)}')

    print('==== Boundary Search Test (using scores with filtering) ====')
    scores = np.random.randn(num).astype(np.float32)
    boundary = train_boundary(data,
                              scores=scores,
                              invalid_value=scores[0],
                              chosen_num_or_ratio=chosen_ratio,
                              verbose_test=True,
                              logger=logger)
    assert boundary.shape == (1, dim)
    print(f'Boundary norm: {np.linalg.norm(boundary)}')
    TEST_FLAG = True
  ###################################
  #### Boundary Search Test Ends ####
  ###################################

  if not TEST_FLAG:
    raise SystemExit('No test has been executed! '
                     'Please use --help to see detailed usage.')
