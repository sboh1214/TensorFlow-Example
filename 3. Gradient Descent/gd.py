import tensorflow as tf

tf.compat.v1.disable_eager_execution()

X_data = [1, 2, 3]
Y_data = [1, 2, 3]

W = tf.Variable(tf.compat.v1.random_uniform([1], minval=-1.0, maxval=1.0))
B = tf.Variable(tf.compat.v1.random_uniform([1], minval=-1.0, maxval=1.0))

X = tf.compat.v1.placeholder(tf.float32, name="X")
Y = tf.compat.v1.placeholder(tf.float32, name="Y")

hypothesis = W * X + B

cost = tf.reduce_mean(tf.square(hypothesis-Y))
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