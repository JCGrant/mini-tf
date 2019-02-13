import os
import mini_tf as tf
# import tensorflow as tf

# Makes comparing with tensorflow easier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with tf.Session() as sess:

    vec = tf.random_uniform(shape=(3, ))
    one = tf.constant(1, dtype=tf.float32)
    out1 = vec + one
    out2 = vec + 2

    print(sess.run(vec))
    print(sess.run(vec))
    print(sess.run((out1, out2)))
    print()


with tf.Session() as sess:

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y

    print(sess.run(z, feed_dict={x: 3, y: 4.5}))
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
    print()


with tf.Session() as sess:

    my_variable = tf.get_variable(
            "my_variable", [1, 2, 3],
            initializer=tf.random_uniform_initializer)

    sess.run(tf.global_variables_initializer())
    # sess.run(my_variable.initializer)

    print(sess.run(my_variable))
    print()


def sequential(*layers):
    def run(x):
        for layer in layers:
            x = layer(x)
        return x
    return run


with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=[None, 3])
    model = sequential(
        tf.layers.Dense(units=10),
        tf.layers.Dense(units=1),
    )
    y = model(x)

    sess.run(tf.global_variables_initializer())

    print(sess.run((x, y), {x: [[1, 2, 3], [4, 5, 6]]}))
    print()


g_1 = tf.Graph()
with g_1.as_default():
    with tf.Session() as sess_1:
        assert sess_1.graph is g_1
        assert tf.get_default_graph() is g_1


g_2 = tf.Graph()
with g_2.as_default():
    with tf.Session() as sess_2:
        assert sess_2.graph is g_2
        assert tf.get_default_graph() is g_2
