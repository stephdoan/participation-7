import os
import pandas as pd
import numpy as np
import re

from sklearn import linear_model
from scipy.signal import find_peaks

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import *
from feature import *

def train_clf(training_data):
    X = training_data[features]
    y = training_data['stream']
    logreg = LogisticRegression()
    logreg.fit(X, y)
    return logreg
