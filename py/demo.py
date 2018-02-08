import load_data as ld
import model
import parser
import numpy as np
from sklearn.metrics import mean_squared_error



def score(y_pred, y):
    # y_pred = self.prediction(X)
    return 1. - np.sqrt(mean_squared_error(y[:, 0], y_pred[:, 0])) / (np.max(y[:, 0]) - np.min(y[:, 0]))

if __name__ == "__main__":

    task, N, dim = ld.random_data()
    task, X, Y = ld.load_school_data()

    # mode = 'train'
    mode = 'infer'


    if mode == 'train':

        FNN = model.model()
        FNN.build_model(28)
        FNN.build_loss()
        FNN.build_training(0.0001)
        FNN.train_model(X,Y)


    if mode == 'infer':

        FNN =  model.model()
        FNN.build_model(28)
        FNN.build_loss()
        y_pred = FNN.infer(X)

        print score(y_pred,Y)
