import os
import pandas as pd
import numpy as np
import re

import scipy as sp
from scipy import fft as fft

import warnings
warnings.filterwarnings('ignore')

def std_df(df, time):
  """
  Takes unix time and standardizes to [time unit] starting from 0.
  time is the time_column
  """
  df[time] = df[time] - np.min(df[time])
  return df

def get_files(lst, category, vpn=True):

  clean_lst = [x for x in lst if not re.search(r"-noisy-", x)]

  vpn_lst = [fp for fp in clean_lst if re.search(r'-vpn-', fp)]
  novpn_lst = [fp for fp in clean_lst if re.search(r'-novpn-', fp)]


  if vpn:
    cat_lst = [file for file in vpn_lst if re.search(category, file)]
  else:
    cat_lst = [file for file in novpn_lst if re.search(category, file)]

  return cat_lst

def chunk_data(df, size):
  """
  returns dict of indices for equal sized samples

  parameters:

  df: dataframe
    network-stats output

  size: int
    period of our observations in a session

  """
  df = std_df(df, 'Time')
  total_chunks = np.floor(df['Time'].max() / size).astype(int)

  max_time_df = df[df['Time'] < (total_chunks * size)]

  max_time_df['Time'] = pd.to_datetime(max_time_df['Time'], unit='s')

  grouped = max_time_df.groupby('Time')[[
    '2->1Bytes',
    '1->2Bytes',
    '2->1Pkts',
    '1->2Pkts']
  ].sum().reset_index()

  sampled = grouped.resample(str(size) +'s', on='Time')
  sampled = list(sampled.indices.values())

  return [grouped, sampled]

def fft_df(df, col):
  df_fft = sp.fft.fft(df[col].values)
  df_amp = np.abs(df_fft)
  df_psd = df_amp ** 2

  df_fft_freq = sp.fft.fftfreq(df_fft.size)

  idx = df_fft_freq > 0

  return pd.DataFrame({
    'freq': df_fft_freq[idx],
    'psd': df_psd[idx]
  })

def get_psd_freq(df, col):
  """
  returns max psd value and corresponding frequency
  """

  fft_temp = fft_df(df, col)

  return fft_temp.iloc[fft_temp['psd'].idxmax()].values
