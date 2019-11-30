import tensorflow as tf

from numpy import savetxt

tf.enable_eager_execution()

with open('tests/fixtures/test_add.txt') as f:
    x, y = f.read().strip().split(' ')

x_tensor = tf.constant(int(x), dtype=tf.int32)
y_tensor = tf.constant(int(y), dtype=tf.int32)

res = x_tensor + y_tensor
savetxt('tests/fixtures/test_add_res.txt', [res])
