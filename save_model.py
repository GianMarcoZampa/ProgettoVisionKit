import tensorflow as tf
import tensorflow.keras as ks
import numpy as np

LOAD_PATH = "weights-improvement-34-0.72.hdf5"
SAVE_PATH = "model_v3.h5"

model = ks.models.load_model(LOAD_PATH)
model.save(SAVE_PATH)
print("Model saved at ", SAVE_PATH)
