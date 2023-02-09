import csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from ..constants import TRAIN, VALID


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


def plot_train_price(csv_file, visualize_folder):
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


def plot_test_price(csv_file, visualize_folder):
    print('Read price csv...')
    figure_settings = {
        'origin': {
            'color': 'black',
            'linestyle': '-'
        },
        'future': {
            'color': 'gray',
            'linestyle': '--'
        }
    }

    price_data = []  # a list of [source_type, price]
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        next(reader, None)
        for row in reader:
            source_type, price = row[0].split('_')[0], float(row[1])
            price_data.append([source_type, price])

    print('Plot...')
    sns.set(
        style='darkgrid',
        rc={'figure.figsize': ((10, 6))}
    )

    # plot lines
    for i in range(len(price_data) - 1):
        source_type = price_data[i + 1][0]
        sns.lineplot(
            x=[i + 1, i + 2], y=[price_data[i][1], price_data[i + 1][1]], 
            color=figure_settings[source_type]['color'],
            linestyle=figure_settings[source_type]['linestyle']
        )
    
    # define labels
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    # define legends
    patches = []
    for source_type, settings in figure_settings.items():
        patches.append(Line2D([0], [0], 
            color=settings['color'], label=source_type,
            linestyle=figure_settings[source_type]['linestyle']
        ))
    plt.legend(handles=patches, loc='upper left')

    visualize_image_file_path = visualize_folder.joinpath(csv_file.with_suffix('.png').name)
    plt.savefig(visualize_image_file_path)
    plt.clf()