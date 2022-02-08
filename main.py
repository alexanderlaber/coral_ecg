#
import numpy as np
#import os
#import datetime
#import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
#tensorboard --logdir logs/fit

input_test = np.load("testdata.npy")
target_test = np.load("testlabel.npy")

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter("model_edgetpu.tflite")
interpreter.allocate_tensors()

# Run an inference
common.set_input(interpreter, input_test[0])
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

print("target: ",target_test[0])
print("pred: ",classes)
# Print the result
#labels = dataset.read_label_file(target_test[0])
#for c in classes:
#  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
# Generate generalization metrics
#
#print(f'Test loss for Keras Leaky ReLU CNN: {score[0]} / Test accuracy: {score[1]}')

