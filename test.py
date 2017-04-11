import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


xy = np.loadtxt('./data/new.csv', delimiter=',', dtype=np.float32)
# hPa_data = MinMaxScaler(np.transpose(xy[:,0:-3]))
# x_data = np.transpose(xy[:, -3:-1])
# x_data = np.append(hPa_data, x_data)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data)
