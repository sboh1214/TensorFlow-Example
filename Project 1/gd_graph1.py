# about 5 hours

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

X_data = [1, 2, 3]
Y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1]))
B = tf.Variable(tf.random_uniform([1]))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = W * X + B

rateList=np.linspace(0.001,0.2,50)
stepList=[]
tempList=[]
for rate in rateList: 
    print("\033[33m"+str(rate))   
    for i in range(50):
        cost = tf.reduce_mean(tf.square(hypothesis-Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
        train_op = optimizer.minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for step in range(3000):
                _, cost_val = sess.run([train_op, cost], feed_dict={X: X_data, Y: Y_data})
                if (cost_val<0.01):
                    tempList.append(step)
                    print(str(i)+" : "+str(step))
                    break
                if (step==2999):
                    tempList.append(3000)
                    print(str(i)+" : NaN")
        stepList.append(np.average(tempList))

f = open('3. Gradient Descent/gd_graph1.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(rateList)
wr.writerow(stepList)
f.close()

print("End")