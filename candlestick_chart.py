import mplfinance as mpf
import yfinance as yf
import pandas as pd


def display_candlestick_chart(ticker, start_date, end_date, n_trading_days=1):
    """
    Display stock market financial data using a candlestick chart.

    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date of the data (format 'YYYY-MM-DD')
    end_date (str): End date of the data (format 'YYYY-MM-DD')
    n_trading_days (int): Number of trading days to group data by (default=1)

    Returns:
    None
    """
    # Download the stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure data is in the correct format
    data.index.name = 'Date'
    data.reset_index(inplace=True)

    # Group the data by n trading days
    data_grouped = data.groupby(pd.Grouper(key='Date', freq=f'{n_trading_days}D')).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Drop any rows with NaN values
    data_grouped.dropna(inplace=True)

    # Create the candlestick chart
    mpf.plot(data_grouped,
             type='candle',
             title=f'{ticker} Candlestick Chart',
             ylabel='Price (USD)',
             ylabel_lower='Volume',
             volume=True,
             style='yahoo',
             figratio=(16, 9),
             figscale=1.2,
             show_nontrading=True,
             datetime_format='%Y-%m-%d'
             )


# Test the function
display_candlestick_chart('AAPL', '2020-01-01', '2022-02-26', n_trading_days=5)
