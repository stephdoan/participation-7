import os
import pandas as pd
import numpy as np
import re

import scipy as sp
from scipy import fft as fft

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

def create_spectral_features(chunk_lst, col):

  chunk_freq = []
  chunk_psd = []

  for chunk in chunk_lst:

    vals = get_psd_freq(chunk, col)

    chunk_freq.append(vals[0])
    chunk_psd.append(vals[1])

  psd_freq_df = pd.DataFrame({
    col+'_psd': chunk_psd,
    col+'_freq': chunk_freq
  })

  return psd_freq_df

def create_features(folder, fp_lst, chunk, labels, video):

  chunk_lst = []

  for data_fp in fp_lst:
    df, idx = chunk_data(pd.read_csv(folder + data_fp), chunk)
    for i in idx:
      chunk_lst.append(df.iloc[i])

  download_bytes = create_spectral_features(chunk_lst, '2->1Bytes')
  upload_bytes = create_spectral_features(chunk_lst, '1->2Bytes')
  download_pkts = create_spectral_features(chunk_lst, '2->1Pkts')
  upload_pkts = create_spectral_features(chunk_lst, '1->2Pkts')

  feature_df =pd.concat([
    download_bytes,
    upload_bytes,
    download_pkts,
    upload_pkts
  ], axis=1)

  if labels:
    return label_data(feature_df, video, labels)

  else:
    return feature_df
