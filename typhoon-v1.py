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


xy = np.loadtxt('./data/x.csv', delimiter=',', dtype=np.float32)
x_data1 = xy[:, 0:25]
x_data2 = xy[:, 25:-1]
y_data = xy[:, [-1]]
#
# print(x_data1.shape)
# print(x_data2.shape)
# print(y_data.shape)
#

# print(x_data.shape, y_data.shape)
x_num = 2
nb_classes = 25

X1 = tf.placeholder(tf.float32, [None, nb_classes])
X2 = tf.placeholder(tf.float32, [None, x_num])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
# print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# print("reshape", Y_one_hot)

W11 = tf.get_variable("W11", shape=[25, 13], initializer=xavier_init(25,13))
W21 = tf.get_variable("W21", shape=[13, 1], initializer=xavier_init(13,1))
W12 = tf.get_variable("W12", shape=[x_num, x_num], initializer=xavier_init(x_num,x_num))
W22 = tf.get_variable("W22", shape=[x_num, x_num], initializer=xavier_init(x_num,x_num))
W31 = tf.get_variable("W31", shape=[3, nb_classes], initializer=xavier_init(3,nb_classes))
W32 = tf.get_variable("W32", shape=[nb_classes, nb_classes], initializer=xavier_init(nb_classes,nb_classes))

b11 = tf.Variable(tf.zeros([13]))
b21 = tf.Variable(tf.zeros([1]))
b12 = tf.Variable(tf.zeros([x_num]))
b22 = tf.Variable(tf.zeros([x_num]))
b31 = tf.Variable(tf.zeros([nb_classes]))
b32 = tf.Variable(tf.zeros([nb_classes]))

_L2 = tf.nn.relu(tf.add(tf.matmul(X1, W11),b11))
# L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(_L2, W21),b21))
# L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(X2, W12),b12))
# L4 = tf.nn.dropout(_L4, dropout_rate)
_L5 = tf.nn.relu(tf.add(tf.matmul(_L4, W22),b22))
X3 = tf.concat([_L3, _L5], 1)

_L6 = tf.nn.relu(tf.add(tf.matmul(X3, W31),b31))
logits = tf.add(tf.matmul(_L6, W32),b32)

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(X3, feed_dict={X1: x_data1, X2: x_data2, Y: y_data}))
    # print(sess.run(tf.shape(X3), feed_dict={X1: x_data1, X2: x_data2, Y: y_data}))

    for step in range(10000):
        sess.run(optimizer, feed_dict={X1: x_data1, X2: x_data2, Y: y_data})
        if 0 == step % 500:
            loss, acc = sess.run([cost, accuracy], feed_dict={X1: x_data1, X2: x_data2, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X1:x_data1, X2: x_data2})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    # print(sess.run([W1,W2,W3,W4], feed_dict={X:x_data, Y:y_data}))
