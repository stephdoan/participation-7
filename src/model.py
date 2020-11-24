from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from feature import *

def train_clf(training_data):
    X = training_data.drop(columns=['video'])
    y = training_data[['video']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=0.3, random_state=0)
    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    clf_scores = classification_report(y_test, y_pred)
    return [logreg, clf_scores]
