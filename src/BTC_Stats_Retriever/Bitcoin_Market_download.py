#Import and download necessary libraries
import subprocess
import  sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])

import yfinance as yf


def downloadBitcoinStats():
    # Download Bitcoin market statistics into Dataframe
    bitcoinDF = yf.Ticker("BTC-USD").history(period="5y", interval="1d")

    # Convert Bitcoin DF into DownloadBTC-USD.csv file located in Datasets Directory
    bitcoinDF.to_csv('../src/DataSets/DownloadBTC-USD.csv')