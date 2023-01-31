import csv
import argparse
from pathlib import Path

from .constants import DATA_FOLDER
from .twstock import __update_codes, Stock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_code", type=str, default="2330", help="the stock code to be fetched")
    parser.add_argument("--start_year_month", type=int, nargs='+', required=True,
                        help="fetch the data from the start year")
    args = parser.parse_args()

    stock_code = args.stock_code
    start_year, start_month = args.start_year_month

    print('Updating stock info...')
    __update_codes()
    print('Done')

    print(f'Initializing {stock_code}...')
    stock = Stock(stock_code)
    print('Done')

    print(f'Fetching from {start_year}/{start_month:02d}...')
    stock.fetch_from(start_year, start_month)
    print('Done')

    year_month2day_prices_dict = {}
    for i, date in enumerate(stock.date):
        year_month = date.strftime("%Y_%m")
        if year_month not in year_month2day_prices_dict:
            year_month2day_prices_dict[year_month] = []

        year_month2day_prices_dict[year_month].append([date.day, 
            stock.open[i], stock.close[i], stock.low[i], stock.high[i]])

    print('Saving price info...')
    data_folder = Path(DATA_FOLDER, stock_code)
    data_folder.mkdir(parents=True, exist_ok=True)
    for year_month, date_prices_list in year_month2day_prices_dict.items():
        date_prices_list.sort(key=lambda x: x[0])
        with open(data_folder.joinpath(f'{year_month}.csv'), 'w', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerow(['date', 'open', 'close', 'low', 'high'])
            for date_prices in date_prices_list:
                writer.writerow([f'{year_month}_{date_prices[0]:02d}'] + date_prices[1:])
    print('Done')


if __name__ == "__main__":
    main()
