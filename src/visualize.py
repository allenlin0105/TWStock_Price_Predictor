import csv
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .constants import MODEL_CKPT_FOLDER, PREDICTION_FOLDER, VISUALIZE_FOLDER


def main():
    parser = ArgumentParser()
    parser.add_argument("--stock_code", type=str, default=2330)
    parser.add_argument("--ckpt_index", type=int, default=0)
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--visualize_epoch", type=int, default=0)
    args = parser.parse_args()

    model_folder = Path(MODEL_CKPT_FOLDER, args.stock_code, f'{args.ckpt_index:03d}')
    target_file_path = model_folder.joinpath(PREDICTION_FOLDER, f'{args.split}_{args.visualize_epoch:03d}.csv')

    print('Read target csv file...')
    data = []
    with open(target_file_path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        next(reader, None)
        for row in reader:
            date, true_price, predict_price = row[0], float(row[1]), float(row[2])
            data.append([date, 'True Price', true_price])
            data.append([date, 'Predict Price', predict_price])
    data.sort(key=lambda x: x[0])

    print('Build dataframe...')
    price_df = pd.DataFrame(data, columns=["Date", "Split", "Price"])

    visualize_folder = model_folder.joinpath(VISUALIZE_FOLDER)
    visualize_folder.mkdir(parents=True, exist_ok=True)

    print('Plot...')
    sns.set(
        style='darkgrid',
        rc={'figure.figsize':(30, 7)}
    )

    plot = sns.lineplot(price_df, x="Date", y="Price", hue="Split", dashes=False)
    locator = mdates.DayLocator(interval=60)
    plot.xaxis.set_major_locator(locator)
    visualize_image_file_path = visualize_folder.joinpath(f'{args.split}_{args.visualize_epoch:03d}.png')
    plt.savefig(str(visualize_image_file_path))


if __name__ == "__main__":
    main()
