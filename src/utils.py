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

def clean_ext_entry(entry, dtype):
  """
  takes an entry, cleans the lists, and stores values in a numpy array.
  helper method for expand_ext

  parameters:
    entry: row entry from [packet_times, packet_sizes, packet_dirs]
    dtype: choose from [float, np.int64, np.float64]

  return:
    array of int or floats
  """
  clean_str = entry[:-1].strip()
  split_str = clean_str.split(';')
  to_type = np.array(split_str).astype(dtype)
  return to_type

def create_ext_df(row, dtype, dummy_y=False, order=False):
  """
  takes in a row (series) from network-stats data and returns a dataframe of extended column entries

  parameters:
    row: row to expand into dataframe
    dtype: choose from [float, np.int64, np.float64]

  return:
    dataframe of collected packet details in a network-stats second
  """

  temp_df = pd.DataFrame(
    {
      'Time': clean_ext_entry(row['packet_times'], dtype),
      'pkt_size': clean_ext_entry(row['packet_sizes'], dtype),
      'pkt_src': clean_ext_entry(row['packet_dirs'], str)
    }
  )

  if dummy_y:
    temp_df['dummy_y'] = np.zeros(len(temp_df))

  if order:
    temp_df['order'] = np.arange(len(temp_df))


  return temp_df

def convert_ms_df(df):
  """
  takes in a network-stats df and a specified second duration.
  convert to milliseconds.
  drop the ip address columns and the aggregate columns
  """
  df_lst = []

  df.apply(lambda x: df_lst.append(create_ext_df(x, np.int64)), axis=1)

  ms_df = pd.concat(df_lst)

  sorted_df = ms_df.sort_values(by=['Time'])

  sorted_df['Time'] = pd.to_datetime(sorted_df['Time'], unit='ms')

  grouped_ms_src = sorted_df.groupby(['Time', 'pkt_src']).agg(
    {'pkt_size':'sum'}).reset_index()

  return grouped_ms_src
