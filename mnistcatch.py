import tensorflow as tf
import tensorflow.keras as keras

def ministdata():
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    train_data_x = x
    train_data_y = y
    test_data_x = x_test
    test_data_y = y_test
    # train_data_len = len(train_data[0])
    # test_data_len = len(test_data[0])
    # print(test_data_len)
    return train_data_x, train_data_y, x_test, y_test

train_x, tarin_y, test_x, test_y = ministdata()
print(train_x[0],tarin_y[0])



