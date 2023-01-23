from argparse import ArgumentParser

import torch

from .utils.main_utils import fix_random_seed, get_model_folder, set_up_logger
from .utils.train_utils import train


def main():
    parser = ArgumentParser()

    parser.add_argument("--data_stock_code_year", type=str, nargs='+', required=True,
                        help="specify the data for training")
    parser.add_argument("--device", type=torch.device, default="cuda",
                        help="the device to train the model")

    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of lstm")
    parser.add_argument("--n_layer", type=int, default=2, help="layer of lstm")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout of lstm")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--n_epoch", type=int, default=10, help="number of training epoch")

    parser.add_argument("--do_valid", action="store_true",
                        help="Split part of the data to do validation")
    args = parser.parse_args()

    args.model_folder = get_model_folder(args.data_stock_code_year[0])
    args.logger = set_up_logger(args.model_folder.joinpath('train_log.log'))
    args.device = args.device if torch.cuda.is_available() else torch.device('cpu')

    fix_random_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
