import os
import pandas as pd
import numpy as np
import re

import scipy as sp
from scipy import signal

import warnings
warnings.filterwarnings('ignore')

from utils import *

def label_data(df, video, labels):
  """
  video is a boolean; True if video and false else
  """

  if not labels:
    return df

  else:
    if video:
      df['video'] = np.ones(len(df))
    else:
      df['video'] = np.zeros(len(df))

  return df

def spectral_features(df, col):

  f, Pxx_den = sp.signal.welch(df[col], fs=2)
  Pxx_den = np.sqrt(Pxx_den)

  peaks = sp.signal.find_peaks(Pxx_den)[0]
  prominences = sp.signal.peak_prominences(Pxx_den, peaks)[0]

  idx_max = prominences.argmax()
  loc_max = peaks[idx_max]

  return [f[loc_max], Pxx_den[loc_max], prominences[idx_max]]

def chunk_data(fp, interval=100, select_col=[
    'Time',
    '1->2Bytes',
    '2->1Bytes',
    '1->2Pkts',
    '2->1Pkts',
    'packet_times',
    'packet_sizes',
    'packet_dirs'
  ]):

  """
  takes in a filepath to the data you want to chunk and feature engineer
  chunks our data into a specified time interval
  each chunk is then turned into an observation to be fed into our classifier
  """
  chunk_feature = []
  chunk_col = [
    'dwl_freq',
    'max_dwl_psd',
    'dwl_peak_prominence',
    'upl_freq',
    'max_upl_psd',
    'upl_peak_prominence'
  ]


  df = std_df(pd.read_csv(fp)[select_col], 'Time')

  total_chunks = np.floor(df['Time'].max() / interval).astype(int)

  for chunk in np.arange(total_chunks):

    start = chunk * interval
    end = (chunk+1) * interval

    temp_df = (df[(df['Time'] >= start) & (df['Time'] < end)])

    preproc = convert_ms_df(temp_df)

    upl_bytes = preproc[preproc['pkt_src'] == '1']
    dwl_bytes = preproc[preproc['pkt_src'] == '2']

    dwl_resampled = dwl_bytes.resample('500ms', on='Time').sum()
    upl_resampled = upl_bytes.resample('500ms', on='Time').sum()

    dwl_psd_freq = spectral_features(dwl_resampled, 'pkt_size')
    upl_psd_freq = spectral_features(upl_resampled, 'pkt_size')

    chunk_feature.append(np.hstack((dwl_psd_freq, upl_psd_freq)))

  #return chunk_lst
  return pd.DataFrame(data=chunk_feature, columns=chunk_col).dropna()

def create_features(folder, fp_lst, chunk, labels, video):

  chunk_col = [
    'dwl_freq',
    'max_dwl_psd',
    'dwl_peak_prominence',
    'upl_freq',
    'max_upl_psd',
    'upl_peak_prominence'
  ]

  chunked_df = pd.DataFrame(columns = chunk_col)

  for data_fp in fp_lst:
    temp_df = chunk_data(folder + data_fp, chunk)
    chunked_df = pd.concat([chunked_df, temp_df])


  if labels:
    return label_data(feature_df, video, labels)

  else:
    return chunked_df
