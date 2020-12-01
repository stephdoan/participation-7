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

    test_params = json.load(open('config/test_data.json'))
    data_params = json.load(open('config/data_params.json'))

    model = pickle.load(open('logreg_model_01122020.sav', 'rb'))

    if 'test' in targets:

        print('Now checking feature target')
        test_chunks = create_features(**test_params)

        print(test_chunks)

        print('Now checking predict target')
        preds = model.predict(test_chunks)

        print(preds)


    if 'feature' in targets:

        chunks = create_features(**data_params)
        chunk_size = data_params['chunk']
        print('Features created: Chunk size = ' + str(chunk_size) + 's')
        print(chunks)


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
