import csv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from .data_utils import read_data, split_data, StockDataset
from ..model import PricePredictor
from ..constants import TRAIN, VALID, PREDICTION_FOLDER


def train(args):
    do_valid = args.do_valid
    logger = args.logger
    device = args.device
    model_folder = args.model_folder

    logger.info(args)

    splits = [TRAIN]
    if do_valid:
        splits.append(VALID)

    raw_dates, raw_prices = read_data(*args.data_stock_code_year)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(raw_prices.reshape(-1, 1))

    splitted_data = {split: split_data(raw_dates, scaled_prices, split, do_valid)
        for split in splits}
    datasets = {split: StockDataset(data)
        for split, data in splitted_data.items()}
    dataloaders = {split: DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == TRAIN)
    ) for split, dataset in datasets.items()}

    model = PricePredictor(args).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = ExponentialLR(optimizer, gamma=0.95)

    prediction_folder = model_folder.joinpath(PREDICTION_FOLDER)
    prediction_folder.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.n_epoch)):
        # Training
        model.train()
        train_loss, train_prices = 0, []

        for input, labels, label_dates in tqdm(dataloaders[TRAIN], desc=f'Epoch {epoch}', position=1):
            optimizer.zero_grad()
            pred_prices = model(input.to(device))
            loss = loss_func(pred_prices, labels.to(device))
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.5, norm_type=2)
            optimizer.step()

            train_loss += loss.item()
            rescaled_labels = scaler.inverse_transform(labels.cpu().detach().numpy())
            rescaled_pred_prices = scaler.inverse_transform(pred_prices.cpu().detach().numpy())
            train_prices += [[date, label_price[0], pred_price[0]]
                for date, label_price, pred_price in zip(label_dates, rescaled_labels, rescaled_pred_prices)]
        # scheduler.step()

        n_train = len(datasets[TRAIN])
        train_loss /= n_train
        logger.info(f'Epoch {epoch:03d} | {TRAIN} | Loss = {train_loss:.5f}')

        train_prices.sort(key=lambda x: x[0])
        with open(prediction_folder.joinpath(f'{TRAIN}_{epoch:03d}.csv'), 'w', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerow(['date', 'label_price', 'predicted_price'])
            for date, label_price, pred_price in train_prices:
                writer.writerow([date, label_price, pred_price])

        if not do_valid:
            continue

        # Validation
        model.eval()
        valid_loss, valid_prices = 0, []

        with torch.no_grad():
            for input, labels, label_dates in tqdm(dataloaders[VALID], desc=f'Epoch {epoch}', position=1):
                pred_prices = model(input.to(device))
                loss = loss_func(pred_prices, labels.to(device))

                valid_loss += loss.item()
                rescaled_labels = scaler.inverse_transform(labels.cpu().detach().numpy())
                rescaled_pred_prices = scaler.inverse_transform(pred_prices.cpu().detach().numpy())
                valid_prices += [[date, label_price[0], pred_price[0]]
                    for date, label_price, pred_price in zip(label_dates, rescaled_labels, rescaled_pred_prices)]

        n_valid = len(datasets[VALID])
        valid_loss /= n_valid
        logger.info(f'Epoch {epoch:03d} | {VALID} | Loss = {valid_loss:.5f}')

        valid_prices.sort(key=lambda x: x[0])
        with open(prediction_folder.joinpath(f'{VALID}_{epoch:03d}.csv'), 'w', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerow(['date', 'label_price', 'predicted_price'])
            for date, label_price, pred_price in valid_prices:
                writer.writerow([date, label_price, pred_price])


def test(args):
    return