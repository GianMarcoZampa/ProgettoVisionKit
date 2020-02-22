import tensorflow as tf
import tensorflow.keras as ks

MODEL_PATH = "model_v1.h5"


model = ks.models.load_model(MODEL_PATH)

model.summary()

print(model.layers[0].name, model.layers[-1].name)
print(model.inputs)
print(model.outputs)
