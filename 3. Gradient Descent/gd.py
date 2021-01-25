import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

w = tf.Variable(tf.random.uniform([1], minval=-1.0, maxval=1.0))
b = tf.Variable(tf.random.uniform([1], minval=-1.0, maxval=1.0))


@tf.function
def hypothesis(x):
    return tf.matmul(w, x) + b


@tf.function
def cost(hypothesis, y):
    return tf.reduce_mean(tf.square(hypothesis - y))


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: X_data, Y: Y_data})
        print(step, cost_val, sess.run(W), sess.run(B))

    print("\n=== Gradient Descent ===")
    print("X: 5  , Y: ", sess.run(hypothesis, feed_dict={X: 5.0}))
    print("X: 2.5, Y: ", sess.run(hypothesis, feed_dict={X: 2.5}))
