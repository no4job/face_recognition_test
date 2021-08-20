import matplotlib.pyplot as plt
from sklearn import metrics

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class CustomClassifier__(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classes_ = np.array([0,1])
    def fit(self, X, y=None):
        pass

    def predict(self, X, y=None):
        pr = np.zeros((X.shape[0], ))
        return pr

    def predict_proba(self, X, y=None):
        pr = np.column_stack((np.zeros(X.shape[0]),X))
        return pr
if __name__ == "__main__":
    X_test = np.arange(100).reshape(100, 1)/100
    y_test = np.concatenate((np.full((30,1), 0, dtype=int),np.full((70,1), 1, dtype=int)))
    test_estimator = CustomClassifier__()
    prb = test_estimator.predict_proba(X_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, prb[:,1], drop_intermediate =False)
    print("\n".join(map(str,list(zip(fpr.tolist(), tpr.tolist(), thresh.tolist())))))
    auc = metrics.auc(fpr, tpr)
    print("AUC:", auc)
    AUC: 0.9871495327102804



    plt.plot(thresh, tpr, label='tpr curve (area = %.2f)' %auc)
    plt.plot(thresh, fpr, label='fpr curve (area = %.2f)' %auc, color='r')
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc, color='g')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    plt.show()