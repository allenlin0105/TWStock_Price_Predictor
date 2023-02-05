from argparse import ArgumentParser

import torch

from .constants import LOG_FILE
from .utils.main_utils import (fix_random_seed, create_new_model_folder,
    get_exist_model_folder, save_params, set_up_logger)
from .utils.train_test_utils import train, test


def main():
    parser = ArgumentParser()

    parser.add_argument("--stock_code", type=str, default="2330", 
                        help="stock code for training and testing")
    parser.add_argument("--device", type=torch.device, default="cuda",
                        help="the device to train and test the model")

    # model
    parser.add_argument("--model_type", type=str, default='lstm', 
                        help='model to use, should be "lstm", "encoder", or "transformer"')
    ## lstm
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of lstm")
    ## encoder, transformer
    parser.add_argument("--d_model", type=int, default=32, help="dimension of transformer")
    parser.add_argument("--n_head", type=int, default=2, help="number of heads of transformer")
    ## both
    parser.add_argument("--n_layer", type=int, default=2, help="layer of lstm or transformer")
    parser.add_argument("--fc_layer", type=int, default=1, help="layer of lstm fc or transformer fc")
    parser.add_argument("--fc_dim", type=int, default=32, help="hidden dimension size of feed forward")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout of model")

    # train
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--n_input_days", type=int, default=20,
                        help="the number of days used as input")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--n_epoch", type=int, default=10, help="number of training epoch")
    parser.add_argument("--loss_func", type=str, default='mse', help='loss function')

    parser.add_argument("--do_train", action='store_true',
                        help="Do training")
    parser.add_argument("--do_valid", action="store_true",
                        help="Split part of the data to do validation")
    parser.add_argument("--train_start_year", type=int, default=2010,
                        help="start year for training")

    parser.add_argument("--do_test", action="store_true",
                        help="Do testing on future timestamps")
    parser.add_argument("--test_ckpt_index", type=int, default=None,
                        help="the model checkpoint index to do testing")
    args = parser.parse_args()

    if args.do_valid:
        assert args.do_train, "Should add --do_train if you add --do_valid"

    args.device = args.device if torch.cuda.is_available() else torch.device('cpu')

    fix_random_seed(args.seed)

    if args.do_train:
        args.model_folder = create_new_model_folder(args.stock_code)
        save_params(vars(args).copy(), args.model_folder.joinpath('params.json'))
        args.logger = set_up_logger(args.model_folder.joinpath(LOG_FILE))
        train(args)

    if args.do_test:
        args.model_folder = get_exist_model_folder(args.stock_code, args.test_ckpt_index)
        test(args)


if __name__ == "__main__":
    main()
