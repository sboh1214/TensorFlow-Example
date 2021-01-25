import tensorflow as tf

a = tf.constant([10], dtype=tf.int32)
b = tf.constant([3], dtype=tf.int32)
c = tf.constant([2], dtype=tf.int32)


@tf.function
def calc():
    return a * b + c


print(calc()[0])
