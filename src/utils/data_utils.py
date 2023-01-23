import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import DATA_FOLDER, TRAIN, VALID


class StockDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.source = torch.Tensor(data['source']).unsqueeze(2)
        self.label = torch.Tensor(data['label']).unsqueeze(1)
        self.predict_date = data['predict_date']

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.label[idx], self.predict_date[idx]


def read_data(stock_code, year):
    data_folder = Path(DATA_FOLDER, stock_code)
    file_paths = [file_path for file_path in sorted(list(data_folder.iterdir()))
        if file_path.name >= f'{year}_01']

    dates, prices = [], []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            next(reader, None)

            for row in reader:
                dates.append(row[0])
                prices.append(float(row[2]))
    return dates, np.array(prices)


def split_data(raw_dates, scaled_prices, split, do_valid=False):
    input_date_period = 20  # use 20 days to predict

    train_ratio = 0.8 if do_valid else 1.0
    if split == TRAIN:
        bound = int(len(raw_dates) * train_ratio)
        dates = raw_dates[:bound]
        close_prices = scaled_prices[:bound, :].reshape(-1)
    elif split == VALID:
        bound = int(len(raw_dates) * train_ratio) - input_date_period
        dates = raw_dates[bound:]
        close_prices = scaled_prices[bound:, :].reshape(-1)
    else:
        raise ValueError('split variable should be "train" or "test"')

    feed_data = {'source': [], 'label': [], 'predict_date': []}
    for i in range(0, len(close_prices) - input_date_period):
        feed_data['source'].append(close_prices[i: i + input_date_period].tolist())
        feed_data['label'].append(close_prices[i + input_date_period])
        feed_data['predict_date'].append(dates[i + input_date_period])
    return feed_data
