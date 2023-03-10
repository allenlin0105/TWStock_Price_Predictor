import csv
from datetime import datetime

import joblib
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from .data_utils import read_data, split_data, StockDataset
from .loss_utils import RMSELoss, MAPELoss
from ..model import LSTMPredictor, EncoderPredictor, TransformerPredictor
from ..constants import (TRAIN, VALID, TEST, 
    PREDICTION_FOLDER, MODEL_FILE, SCALER_FILE, TEST_FILE)


def train(args):
    do_valid = args.do_valid
    logger = args.logger
    device = args.device
    model_folder = args.model_folder

    splits = [TRAIN]
    if do_valid:
        splits.append(VALID)

    raw_dates, raw_prices = read_data(args.stock_code, args.train_start_year)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(raw_prices.reshape(-1, 1))

    splitted_data = {split: split_data(raw_dates, scaled_prices, split, args.n_input_days, do_valid)
        for split in splits}
    datasets = {split: StockDataset(data)
        for split, data in splitted_data.items()}
    dataloaders = {split: DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == TRAIN)
    ) for split, dataset in datasets.items()}

    if args.model_type == 'lstm':
        model = LSTMPredictor(args).to(device)
    elif args.model_type == 'encoder':
        model = EncoderPredictor(args).to(device)
    elif args.model_type == 'transformer':
        model = TransformerPredictor(args).to(device)
    else:
        raise ValueError('The model type is not defined.')
    logger.info(model)

    if args.loss_func == 'mse':
        loss_func = nn.MSELoss()
    elif args.loss_func == 'rmse':
        loss_func = RMSELoss()
    elif args.loss_func == 'mape':
        loss_func = MAPELoss()
    else:
        raise ValueError('The loss function is not defined.')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    model_path = model_folder.joinpath(MODEL_FILE)
    scaler_path = model_folder.joinpath(SCALER_FILE)
    prediction_folder = model_folder.joinpath(PREDICTION_FOLDER)
    prediction_folder.mkdir(parents=True, exist_ok=True)

    best_result = {'epoch': 0, 'loss': 100}

    for epoch in tqdm(range(args.n_epoch)):
        # Training
        model.train()
        for input, labels, label_dates in tqdm(dataloaders[TRAIN], desc=f'Epoch {epoch}', position=1):
            optimizer.zero_grad()
            pred_prices = model(input.to(device))
            loss = loss_func(pred_prices, labels.to(device))
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.5, norm_type=2)
            optimizer.step()
        scheduler.step()

        # Evaluation
        model.eval()
        for split in splits:
            if split == VALID and not do_valid:
                continue

            loss, prices = 0, []
            with torch.no_grad():
                for input, labels, label_dates in tqdm(dataloaders[split], desc=f'Epoch {epoch}', position=1):
                    pred_prices = model(input.to(device))
                    loss += loss_func(pred_prices, labels.to(device)).item()
                    rescaled_labels = scaler.inverse_transform(labels.cpu().detach().numpy())
                    rescaled_pred_prices = scaler.inverse_transform(pred_prices.cpu().detach().numpy())
                    prices += [[date, label_price[0], pred_price[0]]
                        for date, label_price, pred_price in zip(label_dates, rescaled_labels, rescaled_pred_prices)]

            loss /= len(datasets[split])
            logger.info(f'Epoch {epoch:03d} | {split} | Loss = {loss}')

            prices.sort(key=lambda x: x[0])
            with open(prediction_folder.joinpath(f'{split}_{epoch:03d}.csv'), 'w', encoding='utf-8') as fp:
                writer = csv.writer(fp)
                writer.writerow(['date', 'label_price', 'predicted_price'])
                for date, label_price, pred_price in prices:
                    writer.writerow([date, label_price, pred_price])

            # Save model if better
            if (split == VALID or (split == TRAIN and not do_valid)) \
                and loss < best_result['loss']:
                logger.info(f'Save model at epoch {epoch:03d}')
                torch.save(model.state_dict(), model_path)
                best_result['loss'] = loss
                best_result['epoch'] = epoch

    logger.info(f"Best model is saved at epoch {best_result['epoch']:03d} with loss = {best_result['loss']}")

    # Save scaler
    joblib.dump(scaler, scaler_path)


def test(args):
    device = args.device
    model_folder = args.model_folder
    n_input_days = args.n_input_days

    raw_dates, raw_prices = read_data(args.stock_code, datetime.now().year)
    if len(raw_prices) < n_input_days:
        raw_dates, raw_prices = read_data(args.stock_code, datetime.now().year - 1)

    scaler_path = model_folder.joinpath(SCALER_FILE)
    scaler = joblib.load(scaler_path)
    scaled_prices = scaler.fit_transform(raw_prices.reshape(-1, 1))
    input_data = split_data(raw_dates, scaled_prices, TEST, n_input_days)['source']
    original_prices = scaler.inverse_transform(np.array(input_data[0]).reshape(-1, 1)).reshape(-1)
    original_dates = raw_dates[-n_input_days:]

    if args.model_type == 'lstm':
        model = LSTMPredictor(args).to(device)
    elif args.model_type == 'encoder':
        model = EncoderPredictor(args).to(device)
    elif args.model_type == 'transformer':
        model = TransformerPredictor(args).to(device)
    else:
        raise ValueError('The model type is not defined.')
    model_path = model_folder.joinpath(MODEL_FILE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predict_days = 5  # number of days to predict
    predict_prices = []
    for i in range(predict_days): 
        with torch.no_grad():
            reshaped_data = torch.tensor(input_data).unsqueeze(2)
            pred_prices = model(reshaped_data.to(device))
            input_data[0] = input_data[0][1:] + [pred_prices[0][0].cpu().item()]
            rescaled_pred_prices = scaler.inverse_transform(pred_prices.cpu().detach().numpy())
            predict_prices.append(rescaled_pred_prices[0][0])

    prediction_folder = model_folder.joinpath(PREDICTION_FOLDER)
    with open(prediction_folder.joinpath(TEST_FILE), 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerow(['date', 'predict_price'])
        for i, original_price in enumerate(original_prices):
            writer.writerow([f'origin_{i + 1}_{original_dates[i].replace("_", "-")}', original_price])
        for i, predict_price in enumerate(predict_prices):
            writer.writerow([f'future_{i + 1}', predict_price])
