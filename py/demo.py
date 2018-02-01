import load_data as ld
import model
import parser


if __name__ == "__main__":

    task, N, dim = ld.random_data()
    task, X, Y = ld.load_school_data()

    mode = 'train'

    if mode == 'train':

        FNN = model.model()
        FNN.build_model(28)
        FNN.build_loss()
        FNN.build_training(0.001)
        FNN.train_model(X,Y)


    if mode == 'infer':

        FNN =  model.model()
        FNN.build_model(28)
        FNN.build_loss()
        FNN.infer(X)