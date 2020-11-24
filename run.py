import pandas as pd
import numpy as np
import json
import sys
import pickle

sys.path.insert(0, 'src')
from utils import *
from feature import *
from model import *

def main(targets):

    data_params = json.load(open('config/data_params.json'))

    model = pickle.load(open('logreg_model_24112020.sav', 'rb'))

    if 'predict' in targets:

        chunks = create_features(**data_params)

        if data_params['labels']:
            X = chunks.drop(columns=['video'])
            y = chunks['video']

            preds = model.predict(X)

            print(preds)

        else:
            X = chunks
            preds = model.predict(X)
            print(preds)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
