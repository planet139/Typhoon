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


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

xy = np.loadtxt('./data/x.csv', delimiter=',', dtype=np.float32)
# hPa_data = MinMaxScaler(np.transpose(xy[:,0:-3]))
# x_data = np.transpose(xy[:, -3:-1])
# x_data = np.append(hPa_data, x_data)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# print(hPa_data)
# print(hPa_data.shape)
# print(x_data.shape)
# x_data = np.append(hPa_data, x_data)
# print(x_data)
# print(x_data.shape)
# print(tf.Session().run(x_data))
# x_data = np.append(hPa_data, x_data)
# print('x_data',x_data)

# print(x_data.shape, y_data.shape)
x_num = 27
nb_classes = 25

X = tf.placeholder(tf.float32, [None, x_num])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
# print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# print("reshape", Y_one_hot)

W1 = tf.get_variable("W1", shape=[x_num, 50], initializer=xavier_init(26,50))
W2 = tf.get_variable("W2", shape=[50, 50], initializer=xavier_init(50,50))
W3 = tf.get_variable("W3", shape=[50, 50], initializer=xavier_init(50,50))
W4 = tf.get_variable("W4", shape=[50, nb_classes], initializer=xavier_init(50,nb_classes))

b1 = tf.Variable(tf.zeros([50]))
b2 = tf.Variable(tf.zeros([50]))
b3 = tf.Variable(tf.zeros([50]))
b4 = tf.Variable(tf.zeros([nb_classes]))

_L2 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))
# L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(_L2, W2),b2))
# L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(_L3, W3),b3))
# L4 = tf.nn.dropout(_L4, dropout_rate)
logits = tf.add(tf.matmul(_L4, W4), b4)

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if 0 == step % 500:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X:x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
