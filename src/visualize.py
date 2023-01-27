import re
import csv
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .constants import (TRAIN, VALID, LOG_FILE,
    MODEL_CKPT_FOLDER, PREDICTION_FOLDER, VISUALIZE_FOLDER)


def plot_loss(log_file, visualize_folder):
    loss_data = []

    print('Read loss log...')
    delimiter = " | "
    epoch_str, loss_str = "Epoch ", "Loss = "
    with open(log_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            if delimiter not in line:
                continue

            indices = [i for i in range(len(line)) if line.startswith(delimiter, i)]
            epoch = int(line[line.find(epoch_str) + len(epoch_str): indices[0]])
            split = line[indices[0] + len(delimiter): indices[1]]
            loss = float(line[indices[1] + len(delimiter) + len(loss_str):-1])

            loss_data.append([epoch, split, loss])

    print('Build dataframe...')
    loss_df = pd.DataFrame(loss_data, columns=["Epoch", "Split", "Loss"])

    print('Plot...')
    sns.set(
        style='darkgrid',
        rc={'figure.figsize':(9, 6)}
    )

    sns.lineplot(loss_df, x="Epoch", y="Loss", hue="Split", dashes=False)
    visualize_image_file_path = visualize_folder.joinpath('loss.png')
    plt.savefig(visualize_image_file_path)
    plt.clf()


def plot_price(csv_file, visualize_folder):
    print('Read price csv...')
    price_data = []
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        next(reader, None)
        for row in reader:
            date, true_price, predict_price = row[0], float(row[1]), float(row[2])
            price_data.append([date, 'True Price', true_price])
            price_data.append([date, 'Predict Price', predict_price])
    price_data.sort(key=lambda x: x[0])

    print('Build dataframe...')
    price_df = pd.DataFrame(price_data, columns=["Date", "Split", "Price"])

    print('Plot...')
    if TRAIN in csv_file.name:
        figure_size = (27, 9)
        day_interval = 360
    elif VALID in csv_file.name:
        figure_size = (9, 6)
        day_interval = 60

    sns.set(
        style='darkgrid',
        rc={'figure.figsize': figure_size}
    )

    plot = sns.lineplot(price_df, x="Date", y="Price", hue="Split", dashes=False)
    locator = mdates.DayLocator(interval=day_interval)
    plot.xaxis.set_major_locator(locator)

    visualize_image_file_path = visualize_folder.joinpath(csv_file.with_suffix('.png').name)
    plt.savefig(visualize_image_file_path)
    plt.clf()


def main():
    parser = ArgumentParser()
    parser.add_argument("--stock_code", type=str, default=2330)
    parser.add_argument("--ckpt_index", type=int, default=0)

    parser.add_argument("--plot_loss", action="store_true")

    parser.add_argument("--plot_price", action='store_true')
    parser.add_argument("--visualize_epoch", type=int, default=0)
    args = parser.parse_args()

    model_folder = Path(MODEL_CKPT_FOLDER, args.stock_code, f'{args.ckpt_index:03d}')
    visualize_folder = model_folder.joinpath(VISUALIZE_FOLDER)
    visualize_folder.mkdir(parents=True, exist_ok=True)

    if args.plot_loss:
        plot_loss(model_folder.joinpath(LOG_FILE), visualize_folder)

    if args.plot_price:
        for split in [TRAIN, VALID]:
            price_csv_file = model_folder.joinpath(PREDICTION_FOLDER, f'{split}_{args.visualize_epoch:03d}.csv')
            plot_price(price_csv_file, visualize_folder)


if __name__ == "__main__":
    main()
