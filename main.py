#
import pandas as pd

import numpy as np
#import os
#import datetime
#import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
#tensorboard --logdir logs/fit

# Create onehot vector of labels for purpose of multiclassification
def one_hot(lab, classes=18):
    n = len(lab)
    out = np.zeros((n, classes))
    out[range(n), lab] = 1
    return out

# The class loads the train and test data
class DataLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0  # start point for batch counting
        self.sampledata = None
        self.labeldata = None

    # Data is loaded from csv-files. Every row contains one heartbeat and the first 187 columns are the samples of
    # the ecg curve, while the last value is the label.
    def load(self):
        data = pd.read_csv('ekgdatasets/' + str(self._source) + '.csv')
        self.sampledata = data.values[:, :-1]  # datapoints of the ecg curve of one beat
        self.labeldata = data.values[:, -1].astype('int32')  # the last value in the row is
        return self

    def next_batch(self, batch_size):  # split the train data in small batches and convert labels to one hot vector
        beatnumber, sample = self.sampledata.shape
        x, y = self.sampledata[self._i:self._i+batch_size], \
               self.labeldata[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.sampledata)
        x = x.reshape(-1,  sample, 1)
        y = y.reshape(-1, 1).astype('int32')
        y = one_hot(y[:, -1], 18)
        return x, y

# The class names the test and train data
class DataManager(object):
    def __init__(self):
        #self.train = DataLoader("train_mitrauschen_threedatasets").load()  # path for train dataset
        self.test = DataLoader("test").load()  # path for test dataset
data = DataManager()

input_test = data.test.sampledata
input_test = input_test.reshape(-1,  187, 1)
target_test = data.test.labeldata
target_test = target_test.reshape(-1, 1).astype('int32')
target_test = one_hot(target_test[:, -1], 18)

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter("model_edgetpu.tflite")
interpreter.allocate_tensors()

# Run an inference
common.set_input(interpreter, input_test)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(target_test)
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
# Generate generalization metrics
#
#print(f'Test loss for Keras Leaky ReLU CNN: {score[0]} / Test accuracy: {score[1]}')

