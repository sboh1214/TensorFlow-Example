import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.constant([10],dtype=tf.int32)
b = tf.constant([3],dtype=tf.int32)
c = tf.constant([2],dtype=tf.int32)

d = a * b + c

with tf.compat.v1.Session() as sess:
    result = sess.run(d)
    print("\n"+str(result[0])+"\n")