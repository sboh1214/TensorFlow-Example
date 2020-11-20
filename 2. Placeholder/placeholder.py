import tensorflow as tf

tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder(tf.float32, [None, 3])
print(X)

X_data = [[1, 2, 3], [4, 5, 6]]

W = tf.Variable(tf.compat.v1.random_normal([3, 2]))
B = tf.Variable(tf.compat.v1.random_normal([2, 1]))

expr = tf.matmul(X, W) + B

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print("=== X_data ===")
print(X_data)
print("=== W      ===")
print(sess.run(W))
print("=== B      ===")
print(sess.run(B))

print("=== expr   ===")
print(sess.run(expr, feed_dict={X: X_data}))

sess.close()
