import os
import pandas as pd
import numpy as np
import re

from sklearn import linear_model
from scipy.signal import find_peaks

def organize_data(lst):
  """
  takes in a list of file names and the collection condition (vpn or no vpn)

  returns an array of 2 arrays - one contains streaming and the other
  contains no streaming
  """

  stream_lst = []
  nostream_lst = []

  for fp in lst:
    #novpn_match = re.search(r"-novpn-", fp)
    vpn_match = re.search(r"-vpn-", fp)

    if vpn_match:

      streaming = re.findall(r"youtube|vimeo|netflix|amazonprime|disneyplus|espnplus|hbomax|hulu", fp)

      if streaming:
        stream_lst.append(fp)
      else:
        nostream_lst.append(fp)

  return np.array((np.array(nostream_lst), np.array(stream_lst)))

def std_df(df):
  """
  Takes unix time and standardizes to seconds starting at 0.
  """
  df['Time'] = df['Time'] - np.min(df['Time'])
  return df

def get_peaks(df, col='2->1Bytes', min_height=100000):
  """
  gets the peaks in a column. default is set to downloaded bytes columns
  and a minimum height of 100000
  """
  peaks = df.iloc[find_peaks(df[col], height=min_height)[0], :]
  return peaks
