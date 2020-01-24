try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax, Flatten
from tensorflow.keras.layers import MaxPool2D, Dropout

import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from datetime import datetime

%matplotlib inline
import matplotlib.pyplot as plt

"""
  UTILITIES CLASSES FOR BUILDING DEEP NETWORK
"""

#in order to implement a convolutional layer
class OcrConvolutional(Model):
  def __init__(self, in_channels, out_channels, size):
    super().__init__() # setup the model basic functionalities (mandatory)
    initial = tf.random.truncated_normal([size, size, in_channels, out_channels], stddev=0.1)
    self.filters = tf.Variable(initial) # create weights for the filters

  def call(self, x):
    res = tf.nn.conv2d(x, self.filters, 1, padding="SAME")
    return res

#in order to implement a fully connected layer
class OcrFullyConnected(Model):
  def __init__(self, input_shape, output_shape):
    super().__init__() # initialize the model
    self.W = tf.Variable(tf.random.truncated_normal([input_shape, output_shape], stddev=0.1)) # declare weights 
    self.b = tf.Variable(tf.constant(0.1, shape=[1, output_shape]))  # declare biases
    
  def call(self, x):
    res = tf.matmul(x, self.W) + self.b 
    return res

"""
object which represents the entire network with three convolutional layers alternated with three max pool layers,
followed by a fully connected layer regularized with dropout, and finally 
we'll get predictions using again a softmax layer
"""
class OcrDeepModel(Model):
  def __init__(self):
    super().__init__()                            # input shape: (batch, 16, 8, 1)
    self.conv1 = OcrConvolutional(1, 16, 5)       # out shape : (batch, 16, 8, 32)
    self.pool1 = MaxPool2D([2,2])                 # out shape : (batch, 8,  4, 32)
    self.conv2 = OcrConvolutional(16, 32, 4)      # out shape : (batch, 8,  4, 64)
    self.pool2 = MaxPool2D([2,1])                 # out shape : (batch, 4,  4, 64)
    self.conv3 = OcrConvolutional(32, 64, 3)      # out shape : (batch, 4,  4, 64)
    self.pool3 = MaxPool2D([2,2])                 # out shape : (batch, 2,  2, 64)
    self.conv4 = OcrConvolutional(64, 128, 2)     # out shape : (batch, 2,  2, 128)
    self.pool4 = MaxPool2D([2,2])                 # out shape : (batch, 1,  1, 128)
    self.flatten = Flatten()                      # out shape : (batch, 1*1*128)
    self.fc1 = OcrFullyConnected(1*1*128, 256)    # out shape : (batch, 512)
    self.dropout = Dropout(0.5)                   # out shape : unchanged
    self.fc2 = OcrFullyConnected(256, 26)         # out shape : (batch, 26)
    self.softmax = Softmax()                      # out shape : unchanged


  def call(self, x, training=False):
    x = tf.nn.relu(self.conv1(x))
    x = self.pool1(x)
    x = tf.nn.relu(self.conv2(x))
    x = self.pool2(x)
    x = tf.nn.relu(self.conv3(x))
    x = self.pool3(x)
    x = tf.nn.relu(self.conv4(x))
    x = self.pool4(x)

    x = self.flatten(x)
    x = tf.nn.relu(self.fc1(x))

    x = self.dropout(x, training=training) # behavior of dropout changes between train and test
    
    x = self.fc2(x)
    prob = self.softmax(x)
    
    return prob

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

    if((epoch+1) < 5 and train_accuracy_metric.result()*100 < (epoch+1)*10):# no sense to continuosly iterate in these cases
      return

    template = 'Epoch {}, Time {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch+1,
                          datetime.now() - start,
                          train_loss_metric.result(),
                          train_accuracy_metric.result()*100))

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
  
  return test_accuracy_metric.result()*100

"""
  UTILITIES FUNCTIONS & DEFINITIONS
"""

#dictionary to map char into numbers
to_num = { 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,
           'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20,
           'v':21, 'w': 22, 'x': 23, 'y': 24, 'z': 25
          }

#dictionary to map numbers back into chars
to_char = {}
for key in to_num:
  to_char[to_num[key]] = key

#map an array of chars into an array of numbers wrt to the dictionary above
def encode_chars_to_nums(chars):
  nums = []
  for el in chars:
    nums.append( to_num[el] )
  
  return nums

#map an array of nums back into an array of chars wrt to the dictionary above
def decode_nums_to_chars(nums):
  chars = []
  for el in nums:
    chars.append( to_char[el] )
  
  return chars

"""
  Initial setting
"""

tf.random.set_seed(0) # set seed for reproducibility, still not deterministic if a GPU is used

train_data = (pd.read_csv('train-data.csv', header=None)).to_numpy()
train_target = (pd.read_csv('train-target.csv', header=None)).to_numpy()

test_data = (pd.read_csv('test-data.csv', header=None)).to_numpy()
test_target = (pd.read_csv('test-target.csv', header=None)).to_numpy()

#pre-process Xs (fit every letter from a 128 flat array to a 16x8 matrix, convert to tensor shapes and add a third dimension)
train_data = train_data.reshape(train_data.shape[0], 16, 8)
train_data = tf.cast(train_data, tf.float32) / 255.0
#train_data = tf.convert_to_tensor(train_data)
train_data = train_data[..., tf.newaxis]

test_data = test_data.reshape(test_data.shape[0], 16, 8)
test_data = tf.cast(test_data, tf.float32) / 255.0
#test_data = tf.convert_to_tensor(test_data)
test_data = test_data[..., tf.newaxis]

#pre-process Ys (convert to tensor shapes and one hot encoding)
train_target = encode_chars_to_nums(train_target[:,0])
train_target = tf.one_hot(train_target, 26)

test_target = encode_chars_to_nums(test_target[:,0])
test_target = tf.one_hot(test_target, 26)

"""
  Training & Validation (model selection)
"""


possible_learning_rates = [1e-2, 1e-3, 1e-4]
possible_optimizers = [tf.keras.optimizers.SGD, tf.keras.optimizers.Adamax, tf.keras.optimizers.Adam, tf.keras.optimizers.Adadelta, tf.keras.optimizers.Adagrad, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop]
possible_optimizers_names = ["SGD", "Adamax", "Adam", "Adadelta", "Adagrad", "Nadam", "RMSprop"]

selected_learning_rate = 1e-2
selected_optimizer_index = 0

best_accuracy = 0 #select model with best accuracy value

sel_train_data, sel_test_data, sel_train_target, sel_test_target = train_test_split(train_data.numpy(), train_target.numpy(), test_size=0.33, random_state=42)
sel_train_data = tf.convert_to_tensor(sel_train_data)
sel_test_data = tf.convert_to_tensor(sel_test_data)
sel_train_target = tf.convert_to_tensor(sel_train_target)
sel_test_target = tf.convert_to_tensor(sel_test_target)

EPOCHS = 16

sel_train_ds = tf.data.Dataset.from_tensor_slices(
    (sel_train_data, sel_train_target)).shuffle(10000).batch(128)

sel_test_ds = tf.data.Dataset.from_tensor_slices((sel_test_data, sel_test_target)).batch(64)

for i in range(len(possible_optimizers)):
  optimizer = possible_optimizers[i]
  for l_rate in possible_learning_rates:
    print(("\n\nMODEL SELECTION with {0} (learning rate = {1})").format( possible_optimizers_names[i] , l_rate))
    network = OcrDeepModel()

    network_loss = tf.keras.losses.CategoricalCrossentropy()

    network_optimizer = optimizer(learning_rate=l_rate)

    train_loop(EPOCHS, sel_train_ds,  network, network_loss, network_optimizer)

    sel_test_accuracy = test_loop(sel_test_ds, network, network_loss)

    if(sel_test_accuracy > best_accuracy):
      best_accuracy = sel_test_accuracy
      selected_learning_rate = l_rate
      selected_optimizer_index = i

"""
  Final Training
"""
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_data, train_target)).shuffle(10000).batch(128)

test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_target)).batch(64)


# Create an instance of the model
network = OcrDeepModel()

network_loss = tf.keras.losses.CategoricalCrossentropy()

network_optimizer = possible_optimizers[selected_optimizer_index](learning_rate=selected_learning_rate)

print(("\n\nFINAL TRAINING with {0} (learning rate = {1})").format(possible_optimizers_names[selected_optimizer_index],selected_learning_rate))
EPOCHS = 32
train_loop(EPOCHS, train_ds,  network, network_loss, network_optimizer)

"""
  Testing
"""
test_accuracy = test_loop(test_ds, network, network_loss)