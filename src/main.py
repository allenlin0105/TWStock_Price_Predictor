from argparse import ArgumentParser

import torch

from .constants import LOG_FILE
from .utils.main_utils import fix_random_seed, get_model_folder, save_params, set_up_logger
from .utils.train_test_utils import train, test


def main():
    parser = ArgumentParser()

    parser.add_argument("--data_stock_code_year", type=str, nargs='+', required=True,
                        help="specify the data for training")
    parser.add_argument("--device", type=torch.device, default="cuda",
                        help="the device to train the model")

    # model
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of lstm")
    parser.add_argument("--n_layer", type=int, default=2, help="layer of lstm")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout of lstm")
    parser.add_argument("--fc_layer", type=int, default=1, help="layer of fc")

    # train
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--n_epoch", type=int, default=10, help="number of training epoch")
    parser.add_argument("--loss_func", type=str, default='mse', help='loss function')

    parser.add_argument("--do_train", action='store_true',
                        help="Do training")
    parser.add_argument("--do_valid", action="store_true",
                        help="Split part of the data to do validation")
    parser.add_argument("--do_test", action="store_true",
                        help="Do testing on future timestamps")
    args = parser.parse_args()

    if args.do_valid:
        assert args.do_train, "Should add --do_train if you add --do_valid"

    args.model_folder = get_model_folder(args.data_stock_code_year[0])
    args.device = args.device if torch.cuda.is_available() else torch.device('cpu')

    fix_random_seed(args.seed)

    if args.do_train:
        save_params(vars(args).copy(), args.model_folder.joinpath('params.json'))
        args.logger = set_up_logger(args.model_folder.joinpath(LOG_FILE))
        train(args)

    if args.do_test:
        test(args)


if __name__ == "__main__":
    main()
