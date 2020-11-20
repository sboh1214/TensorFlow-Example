import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

X = tf.placeholder(dtype=tf.float32, shape=[28*28, None])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

with tf.name_scope(name="Layer_1"):
    W1 = tf.Variable(tf.random_normal([28,28*28]),dtype=tf.float32, name="W1")

with tf.name_scope(name="Layer_2"):
    W2 = tf.Variable(tf.random_normal([1,28]),dtype=tf.float32,name="W2")


ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])
ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
print("MNIST DateSet Loaded")

with tf.Session() as Sess:
    for batch in enumerate(ds_train):
        image, label = batch["image"], batch["label"]
        X_Data = tf.reshape(tensor=image, shape=[28*28, None], name="X_Data")
        Y_Data = tf.reshape(tensor=label, shape=[None, 1], name="Y_Data")
        Y_Data = tf.cast(Y_Data,dtype=tf.float32)
        print(tf.shape(Y_Data))
        Sess.run()
