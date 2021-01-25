import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
y_data = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1]
])

w= tf.Variable(tf.compat.v1.random_normal(shape=[2,3],dtype=tf.float32),name="Weight")
b = tf.Variable(tf.zeros([3]),name="Bias")

@tf.function
def model(x):
    L = tf.add(tf.matmul(x,w),b)
    L = tf.nn.relu(L)
    return tf.nn.softmax(logits=L, name="Model")

@tf.function
def cost(model, y):
    return tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(model),axis=1))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(name="Optimizer", learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    sess.run([train],feed_dict={X:x_data,Y:y_data})
    if (step+1)%10 == 0:
        print(step+1, sess.run(cost,feed_dict={X:x_data,Y:y_data}))

prediction = tf.argmax(model,axis=1, name="Prediction")
target = tf.argmax(Y,axis=1, name="Target")
print('예측값:', sess.run(prediction, feed_dict={X:x_data,Y:y_data}))
print('예측값:', sess.run(target, feed_dict={X:x_data,Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: %.2f"%sess.run(accuracy*100, feed_dict={X:x_data, Y:y_data}))