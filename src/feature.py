import os
import pandas as pd
import numpy as np
import re

from sklearn import linear_model
from scipy.signal import find_peaks

from utils import *

features = [
  'time_interval_std',
  'avg_down',
  'std_down',
  'avg_up',
  'std_up',
  'packet_ratio'
]

## first feature: packet ratio
def packet_ratio(df):
  """
  return a column of packet ratio in a given second
  """
  return np.mean(df['1->2Pkts'] / df['2->1Pkts'])

## second feature: variance of time between peaks
def time_interval_std(df):
  return np.std(np.diff(df['Time']))

## third feature
def avg_download_bytes(df):
  return np.mean(df['2->1Bytes'])

## fourth feature
def std_download_bytes(df):
  return np.std(df['2->1Bytes'])

## fifth feature:
def avg_upload_bytes(df):
  return np.mean(df['1->2Bytes'])

## sixth feature
def std_upload_bytes(df):
  return np.std(df['1->2Bytes'])

## create features from a chunk of data
def feature_cols(df):

  temp_row = np.array([
    time_interval_std(df),
    avg_download_bytes(df),
    std_download_bytes(df),
    avg_upload_bytes(df),
    std_upload_bytes(df),
    packet_ratio(df)
  ])

  return temp_row
def chunk_data(fp, interval=60, select_col=[
    'Time',
    '1->2Bytes',
    '2->1Bytes',
    '1->2Pkts',
    '2->1Pkts',
    'packet_times',
    'packet_sizes'
  ]):

  """
  takes in a filepath to the data you want to chunk and feature engineer
  chunks our data into a specified time interval
  each chunk is then turned into an observation to be fed into our classifier
  """
  chunk_feature = []
  chunk_col = [
    'time_interval_std',
    'avg_down',
    'std_down',
    'avg_up',
    'std_up',
    'packet_ratio'
  ]

  df = get_peaks(std_df(pd.read_csv(fp)[select_col]))

  total_chunks = np.floor(df['Time'].max() / interval).astype(int)

  for chunk in np.arange(total_chunks):

    start = chunk * interval
    end = (chunk+1) * interval

    temp_df = (df[(df['Time'] >= start) & (df['Time'] < end)])
    chunk_feature.append(feature_cols(temp_df))

  return pd.DataFrame(data=chunk_feature, columns=chunk_col).dropna()
