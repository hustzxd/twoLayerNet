import scipy.io as scio
import numpy as np


class ReadMat:
    def __init__(self, mat_file):
        """
        :param mat_file:
        10000 (xa ya) (xb yb)
        1000 (xa_test ya_test) (xb_test yb_test)
        """
        data = scio.loadmat(mat_file)
        self.xa = data['xa']
        self.ya = data['ya']
        self.xb = data['xb']
        self.yb = data['yb']
        self.xa_test = data['xa_test']
        self.ya_test = data['ya_test']
        self.xb_test = data['xb_test']
        self.yb_test = data['yb_test']

    def getData(self):
        num_positive = self.xa.shape[1]
        num_train = self.xa.shape[1] + self.xb.shape[1]
        X = np.zeros((num_train, 2))
        y = np.zeros(num_train, dtype=int)
        X[0:num_positive, 0] = self.xa
        X[num_positive:, 0] = self.xb
        X[0:num_positive, 1] = self.ya
        X[num_positive:, 1] = self.yb
        y[0:num_positive] = 0
        y[num_positive:] = 1

        num_positive = self.xa_test.shape[1]
        num_val = self.xa_test.shape[1] + self.xb_test.shape[1]
        X_val = np.zeros((num_val, 2))
        y_val = np.zeros(num_val, dtype=int)
        X_val[0:num_positive, 0] = self.xa_test
        X_val[num_positive:, 0] = self.xb_test
        X_val[0:num_positive, 1] = self.ya_test
        X_val[num_positive:, 1] = self.yb_test
        y_val[0:num_positive] = 0
        y_val[num_positive:] = 1

        return X, y, X_val, y_val
