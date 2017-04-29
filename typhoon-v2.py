import tensorflow as tf
import numpy as np


tf.set_random_seed(777)
def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)


xy = np.loadtxt('train.csv', delimiter=',', dtype=np.float32)
x_training_data = xy[:-5, 0:-1]
y_training_data = xy[:-5, [-1]]
x_testing_data = xy[-5:, 0:-1]
y_testing_data = xy[-5:, [-1]]

x_arg = 4
location = 400

X = tf.placeholder(tf.float32, [None, x_arg])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, location)
Y_one_hot = tf.reshape(Y_one_hot, [-1, location])

W1 = tf.get_variable("W1", shape=[x_arg, 100], initializer=xavier_init(x_arg,100))
W2 = tf.get_variable("W2", shape=[100, 200], initializer=xavier_init(100,200))
W3 = tf.get_variable("W3", shape=[200, 300], initializer=xavier_init(200,300))
W4 = tf.get_variable("W4", shape=[300, location], initializer=xavier_init(300,location))

b1 = tf.Variable(tf.zeros([100]))
b2 = tf.Variable(tf.zeros([200]))
b3 = tf.Variable(tf.zeros([300]))
b4 = tf.Variable(tf.zeros([location]))

L2 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W2),b2))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W3),b3))
logits = tf.add(tf.matmul(L4, W4), b4)
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(40001):
            sess.run(optimizer, feed_dict={X: x_training_data, Y: y_training_data})
            if 0 == step % 5000:
                loss, acc = sess.run([cost, accuracy], feed_dict={X: x_training_data, Y: y_training_data})
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

        pred = sess.run(prediction, feed_dict={X:x_testing_data})

        # y_data: (N,1) = flatten => (N, ) matches pred.shape
        for p, y in zip(pred, y_testing_data.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
