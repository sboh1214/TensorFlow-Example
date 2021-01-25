import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[28*28, None])
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])

with tf.compat.v1.name_scope(name="Layer_1"):
    w1 = tf.Variable(tf.random.normal([28,28*28]),dtype=tf.float32, name="W1")

with tf.compat.v1.name_scope(name="Layer_2"):
    w2 = tf.Variable(tf.random.normal([1,28]),dtype=tf.float32,name="W2")


ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])
ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
print("MNIST DateSet Loaded")

with tf.compat.v1.Session() as session:
    for batch in enumerate(ds_train):
        image, label = batch["image"], batch["label"]
        X_Data = tf.reshape(tensor=image, shape=[28*28, None], name="X_Data")
        Y_Data = tf.reshape(tensor=label, shape=[None, 1], name="Y_Data")
        Y_Data = tf.cast(Y_Data,dtype=tf.float32)
        print(tf.shape(Y_Data))
        session.run()
