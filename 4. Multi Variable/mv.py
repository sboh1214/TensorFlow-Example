import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

X_Data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
Y_Data = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1]
])

X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,2], name="X")
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,3], name="Y")

W = tf.Variable(tf.compat.v1.random_normal(shape=[2,3],dtype=tf.float32),name="Weight")
B = tf.Variable(tf.zeros([3]),name="Bias")

L = tf.add(tf.matmul(X,W),B)
L = tf.nn.relu(L)

Model = tf.nn.softmax(logits=L, name="Model")
Cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.compat.v1.log(Model),axis=1))

GDOptimizer = tf.compat.v1.train.GradientDescentOptimizer(name="Optimizer", learning_rate=0.1)
Train = GDOptimizer.minimize(Cost)

Sess = tf.compat.v1.Session()
Sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    Sess.run([Train],feed_dict={X:X_Data,Y:Y_Data})
    if (step+1)%10 == 0:
        print(step+1, Sess.run(Cost,feed_dict={X:X_Data,Y:Y_Data}))

Prediction = tf.argmax(Model,axis=1, name="Prediction")
Target = tf.argmax(Y,axis=1, name="Target")
print('예측값:', Sess.run(Prediction, feed_dict={X:X_Data,Y:Y_Data}))
print('예측값:', Sess.run(Target, feed_dict={X:X_Data,Y:Y_Data}))

IsCorrect = tf.equal(Prediction, Target)
Accuracy = tf.reduce_mean(tf.cast(IsCorrect, tf.float32))
print("정확도: %.2f"%Sess.run(Accuracy*100, feed_dict={X:X_Data, Y:Y_Data}))