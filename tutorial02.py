# 학습 모델을 만드는 방법
# 분류(classification) 모델 ==> 거의 로지스틱 펑션을 쓴다.


import tensorflow as tf
import numpy as np

# data 불러오기
# xy = np.loadtxt()

#x와 y를 각 변수에 담기
input_data = [[1,5,3,7,8,10,12],
              [5,8,10,3,9,10,1]]
label_data = [[0,0,0,1,0],
              [1,0,0,0,0]]

# 상수로 지정
INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5

Learning_Rate = 0.05

#담을 그릇
x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])

tensor_map = {x:input_data, y_:label_data}

# 파라메타
W1 = tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32)
W2 = tf.Variable(tf.truncated_normal([HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32)
W3 = tf.Variable(tf.truncated_normal([HIDDEN2_SIZE, CLASSES]), dtype=tf.float32)

b1= tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32)
b2= tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32)
b3= tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32)

# 모델, sigmoid
hiddenLayer1 = tf.sigmoid(tf.matmul(x, W1) + b1)
hiddenLayer2 = tf.sigmoid(tf.matmul(hiddenLayer1, W2) + b2)
y = tf.sigmoid(tf.matmul(hiddenLayer2, W3) + b3)

# cost function ==> cross entropy
cost = tf.reduce_mean(-y_*tf.log(y)-(1-y_)*tf.log(1-y))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

# 트레이닝
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, loss = sess.run([train, cost], feed_dict=tensor_map)
        if(0 == i%100):
            print("step: ", i)
            print("loss: ", loss)
    sess.close()
