import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(777)
def xavier_init(n_inputs, n_outputs):
    init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)

<<<<<<< HEAD
X_ARG = 3
LOCATION = 400
TEST_SET = 4

xy = np.loadtxt('train.csv', delimiter=',', dtype=np.float32)
=======
X_ARG = 4
LOCATION = 400
TEST_SET = 5

xy = np.loadtxt('train.csv', delimiter=',', dtype=np.float32)

# 표준화
for i in range(0,4):
    xy[:,i]=np.divide(xy[:,i] - np.mean(xy[:,i]), np.std(xy[:,i]))
>>>>>>> origin/master
x_training_data = xy[:-TEST_SET, :-1]
y_training_data = xy[:-TEST_SET, [-1]]
x_testing_data = xy[-TEST_SET:, :-1]
y_testing_data = xy[-TEST_SET:, [-1]]

# delete arr, obj, axis(row=0, col=1)
<<<<<<< HEAD
# x_training_data=np.delete(x_training_data, 0, 1)
=======
# x_training_data=np.delete(x_training_data, 1, 1)
>>>>>>> origin/master

X = tf.placeholder(tf.float32, [None, X_ARG])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, LOCATION)
Y_one_hot = tf.reshape(Y_one_hot, [-1, LOCATION])

with tf.name_scope("Layer1") as scope:
    W1 = tf.get_variable("W1", shape=[X_ARG, 100], initializer=xavier_init(X_ARG,100))
    b1 = tf.Variable(tf.zeros([100]))
    L1 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))

    w1_hist = tf.summary.histogram("weight1", W1)
    b1_hist = tf.summary.histogram("bias1", b1)
    l1_hist = tf.summary.histogram("layer1", L1)

with tf.name_scope("Layer2") as scope:
    W2 = tf.get_variable("W2", shape=[100, 200], initializer=xavier_init(100,200))
    b2 = tf.Variable(tf.zeros([200]))
    L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2),b2))

    w2_hist = tf.summary.histogram("weight2", W2)
    b2_hist = tf.summary.histogram("bias2", b2)
    l2_hist = tf.summary.histogram("layer2", L2)

with tf.name_scope("Layer3") as scope:
    W3 = tf.get_variable("W3", shape=[200, LOCATION], initializer=xavier_init(200,LOCATION))
    b3 = tf.Variable(tf.zeros([LOCATION]))
    logits = tf.add(tf.matmul(L2, W3), b3)
    hypothesis = tf.nn.softmax(logits)

    w3_hist = tf.summary.histogram("weight3", W3)
    b3_hist = tf.summary.histogram("bias3", b3)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

<<<<<<< HEAD
=======

>>>>>>> origin/master
# Cross entropy cost/loss
with tf.name_scope("cost") as scope:
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)

    cost_summ = tf.summary.scalar("cost_scalar", cost)
    cost_hist = tf.summary.histogram("cost_hist", cost)

with tf.name_scope("train") as scope:
<<<<<<< HEAD
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0007).minimize(cost)
=======
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0008).minimize(cost)

>>>>>>> origin/master

# Accuracy compute
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
<<<<<<< HEAD
=======

>>>>>>> origin/master
temp = []
for i in range(7):
    temp.append([])

with tf.Session() as sess:
<<<<<<< HEAD
        # merged_summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("./logs/x")
        # writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        for step in range(6001):
            # summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x_training_data, Y: y_training_data})
            sess.run(optimizer, feed_dict={X: x_training_data, Y: y_training_data})
            # writer.add_summary(summary, global_step=step)
            if 0 == step % 100:
=======
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/x")
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        for step in range(40001):
            summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x_training_data, Y: y_training_data})
            writer.add_summary(summary, global_step=step)
            if 0 == step % 5000:
>>>>>>> origin/master
                loss, acc = sess.run([cost, accuracy], feed_dict={X: x_training_data, Y: y_training_data})
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

        pred = sess.run(prediction, feed_dict={X:x_testing_data})

        # y_data: (N,1) = flatten => (N, ) matches pred.shape
        for p, y in zip(pred, y_testing_data.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
<<<<<<< HEAD
            temp[0].append(p)

for i in range(TEST_SET):
    # +24H Y'_예측값
    temp[1].append(120.5+int(temp[0][i]%20))
    temp[2].append(39.5-int(int(temp[0][i])/20))
    # +24H Y
    temp[3].append(120.5+int(y_testing_data[i]%20))
    temp[4].append(39.5-int(int(y_testing_data[i])/20))
    # 0H
    temp[5].append(120.5+(x_testing_data[i][-1]%20))
    temp[6].append(39.5-int(int(x_testing_data[i][-1])/20))

# Show
f, ((sp1,sp2),(sp3,sp4)) = plt.subplots(2, 2, figsize=(4, 4))
img = plt.imread("map.png")

sp = [sp1,sp2,sp3,sp4]
number = 0
for i in sp:
    i.scatter(temp[5][number],temp[6][number], label='0H')
    i.scatter(temp[1][number],temp[2][number], alpha=.5, s=130, marker=(5,1), label='+24H Y\'')
    i.scatter(temp[3][number],temp[4][number], label='+24H Y')
    i.imshow(img, extent=[120, 140, 20, 40])
    i.set_xticks(np.arange(120, 141, 10))
    i.set_yticks(np.arange(20, 41, 10))
    i.legend(loc=0)
    number += 1

=======
            # temp[0].append(p)


for i in range(TEST_SET):
    # Y'
    temp[1].append(120.5+int(temp[0][i]%20))
    temp[2].append(39.5-int(int(temp[0][i])/20))
    # Y
    temp[3].append(120.5+int(y_testing_data[i]%20))
    temp[4].append(39.5-int(int(y_testing_data[i])/20))

# Show
fig = plt.figure(0)
img = plt.imread("map.png")
ax = fig.gca()
ax.imshow(img, extent=[120, 140, 20, 40])
ax.set_xticks(np.arange(120, 141, 10))
ax.set_yticks(np.arange(20, 41, 10))

plt.scatter(temp[3], temp[4], label='Y', color='b', marker='*', s=100)
plt.scatter(temp[1], temp[2], label='Y\'', color='r', marker='o', s=70)
>>>>>>> origin/master
plt.xlim((120,140))
plt.ylim((20,40))
plt.grid()
plt.show()
