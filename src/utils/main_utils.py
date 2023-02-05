import json
import random
import logging
from pathlib import Path

import torch
import numpy as np

from ..constants import MODEL_CKPT_FOLDER


def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_new_model_folder(stock_code):
    model_folder = Path(MODEL_CKPT_FOLDER, stock_code)
    model_folder.mkdir(parents=True, exist_ok=True)

    existed_ckpt_indices = [int(file_path.name) for file_path in sorted(list(model_folder.iterdir()))]
    picked_ckpt_index = 0
    for existed_ckpt_index in existed_ckpt_indices:
        if picked_ckpt_index < existed_ckpt_index:
            break
        picked_ckpt_index += 1

    model_folder = model_folder.joinpath(f'{picked_ckpt_index:03d}')
    model_folder.mkdir()

    return model_folder

def get_exist_model_folder(stock_code, ckpt_index):
    model_folder = Path(MODEL_CKPT_FOLDER, stock_code)
    if ckpt_index is None:
        existed_ckpt_indices = [int(file_path.name) for file_path in model_folder.iterdir()]
        ckpt_index = max(existed_ckpt_indices)
    model_folder = model_folder.joinpath(f'{ckpt_index:03d}')
    return  model_folder


def save_params(saved_object, params_file):
    for key, value in saved_object.items():
        if isinstance(value, torch.device) or isinstance(value, Path):
            saved_object[key] = str(saved_object[key])

    # remove other models' parameters
    if saved_object['model_type'] == 'lstm':
        del saved_object['d_model']
        del saved_object['n_head']
    elif saved_object['model_type'] == 'encoder' or saved_object['model_type'] == 'transformer':
        del saved_object['hidden_size']

    with open(params_file, "w", encoding='utf-8') as fp:
        json.dump(saved_object, fp, indent=4)


def set_up_logger(log_file):
    logger = logging.getLogger(name='train_log')
    logger.setLevel(logging.INFO)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Logging to %s', str(log_file))

    return logger
