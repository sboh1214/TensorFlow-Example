import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np

rateMat = []
stepList = []
costList = []

f = open('Project 1/gd_graph2.csv', 'r', encoding='utf-8', newline='')
data = list(csv.reader(f))

for i in range(len(data[0])):
    rate = float(data[0][i])
    rateMat.append([1/(rate**3),1/(rate**2),1/rate,1])
for i in range(len(data[1])):
    stepList.append([float(data[1][i])])

X_Mat = tf.placeholder(dtype=tf.float32,shape=[None,4],name="X")
Y = tf.placeholder(dtype=tf.float32,shape=[None,1],name="Y")
Rate = tf.placeholder(dtype=tf.float32,shape=[1],name="Rate")

W_Mat = tf.Variable(tf.random_normal(shape=[4,1]), name="W")

H = tf.matmul(X_Mat,W_Mat)

Cost = tf.reduce_mean(tf.square(Y-H))
optimizer = tf.train.AdamOptimizer(learning_rate=5)
train = optimizer.minimize(Cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    cost,_ = sess.run([Cost,train],feed_dict={X_Mat:rateMat,Y:stepList})
    print(step, " ", cost)
    costList.append(cost)

W = sess.run([W_Mat])
print(W[0][1]," ",W[0][1]," ",W[0][2]," ",W[0][3])

plt.figure()
plt.plot(costList)
plt.show()