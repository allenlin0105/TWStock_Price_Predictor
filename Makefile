fetch-data:
	python3 -m src.data_fetcher --stock_code 2885 \
		--start_year_month 2010 1

train-model:
	python3 -m src.main --data_stock_code_year 2885 2010 \
		--batch_size 64 --n_epoch 50 --hidden_size 32 \
		--do_valid
