import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax, Flatten

tf.random.set_seed(0) # set seed for reproducibility, still not deterministic if a GPU is used

mnist = tf.keras.datasets.mnist

(data_train, targets_train), (data_test, targets_test) = mnist.load_data()



plt.gray()
plt.matshow(255 - data_train[0]) # 255 - x simply inverts the fading direction of the image
plt.show()


x_train = tf.cast(data_train, tf.float32) / 255.0
x_test = tf.cast(data_test, tf.float32) / 255.0

# Add a fourth dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

y_train = tf.one_hot(targets_train, 10)
y_test = tf.one_hot(targets_test, 10)




class MnistPerceptron(Model): # inherit from Model
  def __init__(self):
    super().__init__() # initialize Model
    self.flatten = Flatten() # used to flatten pixels of images
    self.W = tf.Variable(tf.zeros([784, 10])) # declare weights with shape :(748, 10)
    self.b = tf.Variable(tf.zeros([1, 10]))   # declare biases, with shape :(1, 10)
    self.softmax = Softmax()
    

  def call(self, x,training=False): 
    # the  training argument is unused in this model, we will need it later
                                                  
    flat = self.flatten(x) # flatten images   
            
    multiplied = tf.matmul(flat, self.W) # matmul, output shape : (batch, 10)
    # we can equivalently do:
    #multiplied = tf.transpose(tf.matmul(tf.traspose(self.W), tf.traspose(flat)))

    fwded = multiplied + self.b # broadcast self.b to (batch, 10) and add   

    prob = self.softmax(fwded) # softmax              
    return prob

# Create an instance of the model
perceptron = MnistPerceptron()


perceptron_loss = tf.keras.losses.CategoricalCrossentropy()

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(100)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

perceptron_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

train_loss_metric = tf.keras.metrics.Mean()
train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

test_loss_metric = tf.keras.metrics.Mean()
test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()


def train_step(images, labels, model, loss_fn, optimizer):
  with tf.GradientTape() as tape: # all the operations within this scope will be recorded in tape
    predictions = model(images, training=True)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss_metric(loss)
  train_accuracy_metric(labels, predictions)


def train_loop(epochs, train_ds, model, loss_fn, optimizer):
    for epoch in range(epochs):
        # reset the metrics for the next epoch
        train_loss_metric.reset_states()
        train_accuracy_metric.reset_states()

        start = datetime.now() # save start time 
        for images, labels in train_ds:
            train_step(images, labels, model, loss_fn, optimizer)

        template = 'Epoch {}, Time {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch+1,
                            datetime.now() - start,
                            train_loss_metric.result(),
                            train_accuracy_metric.result()*100))

EPOCHS = 10
train_loop(EPOCHS, train_ds, perceptron, perceptron_loss, perceptron_optimizer)


def test_step(images, labels, model, loss_fn):
  predictions = model(images, training=False)
  t_loss = loss_fn(labels, predictions)

  test_loss_metric(t_loss)
  test_accuracy_metric(labels, predictions)

def test_loop(test_ds, model, loss_fn):
# reset the metrics for the next epoch
  test_loss_metric.reset_states()
  test_accuracy_metric.reset_states()
 
  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels, model, loss_fn)

  template = 'Test Loss: {}, Test Accuracy: {}'
  print(template.format(test_loss_metric.result(),
                        test_accuracy_metric.result()*100))

test_loop(test_ds, perceptron, perceptron_loss)