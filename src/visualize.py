from argparse import ArgumentParser

from .constants import (TRAIN, VALID, LOG_FILE, TEST_FILE,
    PREDICTION_FOLDER, VISUALIZE_FOLDER)
from .utils.main_utils import get_exist_model_folder
from .utils.plot_utils import plot_loss, plot_train_price, plot_test_price


def main():
    parser = ArgumentParser()
    parser.add_argument("--stock_code", type=str, default=2330)
    parser.add_argument("--ckpt_index", type=int, default=None)

    parser.add_argument("--plot_loss", action="store_true")

    parser.add_argument("--plot_train_price", action='store_true')
    parser.add_argument("--visualize_epoch", type=int, default=0)

    parser.add_argument("--plot_test_price", action='store_true')
    args = parser.parse_args()

    model_folder = get_exist_model_folder(args.stock_code, args.ckpt_index)
    visualize_folder = model_folder.joinpath(VISUALIZE_FOLDER)
    visualize_folder.mkdir(parents=True, exist_ok=True)

    if args.plot_loss:
        plot_loss(model_folder.joinpath(LOG_FILE), visualize_folder)

    if args.plot_train_price:
        for split in [TRAIN, VALID]:
            price_csv_file = model_folder.joinpath(PREDICTION_FOLDER, f'{split}_{args.visualize_epoch:03d}.csv')
            if price_csv_file.exists():
                plot_train_price(price_csv_file, visualize_folder)

    if args.plot_test_price:
        plot_test_price(model_folder.joinpath(PREDICTION_FOLDER, TEST_FILE), visualize_folder)


if __name__ == "__main__":
    main()
