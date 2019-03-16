import tensorflow as tf

#### Important
# a = tf.constant("Hello World")
# print(a)

#### Important
# pi = tf.constant(3.14, name="pi")
# r = tf.placeholder(tf.float32, name="r")
#
# a = pi * r * r
#
# graph = tf.get_default_graph()
# print(graph.get_operations())
#
# with tf.Session() as sess:
#     print(a.eval(feed_dict={r: [5]}))
#     print(sess.run(a, feed_dict={r: [5]}))

g = tf.Graph()
print("g.get_operations()", g.get_operations())
a = tf.placeholder(dtype=tf.float32, name="a")


def method_x():
    b = tf.placeholder(dtype=tf.float32, name="b")
    output = a * b
    print("output", output)
    # g = tf.get_default_graph()
    # print("g.get_operations():", g.get_operations())
    with tf.Session() as sess:
        x = sess.run(output, feed_dict={a: 4, b: 3})
        print("x:", x)
        return x


def method_y(output):
    c = tf.placeholder(dtype=tf.float32, name="c")
    d = tf.placeholder(dtype=tf.float32, name="d")
    o = output * c * d
    print("output_y", o)
    # g = tf.get_default_graph()
    # print("g.get_operations()_y:", g.get_operations())
    with tf.Session() as sess:
        y = sess.run(o, feed_dict={c: 4, d: 3})
        z = o.eval(feed_dict={c: 4, d: 3})
        print("y:", y)
        print("z:", z)


op = method_x()
method_y(op)
