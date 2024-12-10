import tensorflow as tf
x = [[1, 2, 3],
     [1, 2, 3]]

xx = tf.cast(x,tf.float32)

mean_all = tf.reduce_mean(xx)

print(mean_all)