MODEL=transformer

fetch-data:
	python3 -m src.data_fetcher --stock_code 2330 \
		--start_year_month 2010 1

train-with-valid:
	python3 -m src.main --stock_code 2330 --train_start_year 2010 \
		--do_train --do_valid \
		--model_type ${MODEL} --d_model 64 --n_head 4 --n_layer 2 --fc_layer 1 \
		--batch_size 128 --n_epoch 100

train-full-data:
	python3 -m src.main  --stock_code 2330 --train_start_year 2010 \
		--do_train \
		--model_type ${MODEL} --d_model 64 --n_head 4 --n_layer 2 --fc_layer 1 \
		--batch_size 128 --n_epoch 100

test-model:
	python3 -m src.main --stock_code 2330 \
		--do_test \
		--model_type ${MODEL} --d_model 64 --n_head 4 --n_layer 2 --fc_layer 1

visualize-loss:
	python3 -m src.visualize --stock_code 2330 --plot_loss

visualize-train-price:
	python3 -m src.visualize --stock_code 2330 \
		--plot_train_price --visualize_epoch 2

visualize-test-price:
	python3 -m src.visualize --stock_code 2330 \
		--plot_test_price

visualize-all: visualize-loss visualize-train-price visualize-test-price

all: train-with-valid test-model visualize-all
