import tensorflow as tf

x_data = [[1, 2, 3], [4, 5, 6]]

w = tf.Variable(tf.random.normal([3, 2]))
b = tf.Variable(tf.random.normal([2, 1]))


@tf.function
def forward(x):
    return tf.matmul(x, w) + b


print("=== x_data ===")
print(x_data)
print("=== w      ===")
print(w)
print("=== b      ===")
print(b)

print("=== expr   ===")
for x in x_data:
    print(forward(x))
