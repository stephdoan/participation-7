import pandas as pd
import numpy as np
import json
import sys

sys.path.insert(0, 'src')
from utils import *
from feature import *
from train import *

def main(targets):

    data_fp_params = json.load(open('config/data-fp.json'))

    trained = train_clf(pd.read_csv('data/training.csv'))
    logreg_clf = trained[0]
    clf_report = trained[1]

    if 'predict' in targets:

        chunked = chunk_data(**data_fp_params)

        pred_lst = []

        for i in np.arange(len(chunked)):

          pred_i = logreg_clf.predict(chunked.loc[i].values.reshape(1, -1))

          if (pred_i == 1):
              pred_lst.append('yes')

          else:
              pred_lst.append('no')
        sec = data_fp_params['interval']
        time_intervals = [(i*sec, (i+1)*sec) for i in np.arange(len(pred_lst))]

        print(pd.DataFrame({
                'time_intervals': time_intervals,
                'stream?': pred_lst
        })
        )

    if 'scores' in targets:
        print(clf_report)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
