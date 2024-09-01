import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd


def display_boxplot_chart(ticker, start_date, end_date, n_trading_days=1):
    """
    Display stock market financial data using a boxplot chart.

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
    # For the boxplot, we'll create a list of closing prices for each n-day window
    data_grouped = data.groupby(pd.Grouper(key='Date', freq=f'{n_trading_days}D'))['Close'].apply(list)

    # Prepare data for boxplot
    boxplot_data = [prices for prices in data_grouped if len(prices) > 1]

    # Create the boxplot chart
    plt.figure(figsize=(12, 8))
    plt.boxplot(boxplot_data, labels=[f'{n_trading_days}D' for _ in range(len(boxplot_data))])
    plt.title(f'{ticker} Boxplot Chart')
    plt.xlabel('Time Interval')
    plt.ylabel('Closing Price (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


# Test the function
display_boxplot_chart('AAPL', '2020-01-01', '2022-02-26', n_trading_days=5)
