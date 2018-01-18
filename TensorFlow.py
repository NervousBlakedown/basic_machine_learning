#Import TensorFlow into Jupyter Notebook program; opens new browser tab.
import tensorflow as tf
hello = tf.constant("hello world")
sess = tf.session()
print(sess.run(hello))
