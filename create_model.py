import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import datetime as dt

NUM_CLASSES = 10
SHAPE = (32, 32, 3)
EPOCHS = 50
BATCH_SIZE = 128
SAVE_PATH = "model_v1.h5"


def get_data():
    return ks.datasets.cifar10.load_data()


def scale(images):
    x_scaled = []

    # Scale rgb value in range (0,1)
    for i in range(images.shape[0]):
        x_scaled.append(images[i] * 1 / 255)

    x_scaled = np.array(x_scaled)

    return x_scaled


def create_model():
    model = ks.models.Sequential()
    model.add(ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=SHAPE,
                               name="input_image"))
    model.add(ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))

    model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same'))
    model.add(ks.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same'))

    model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(ks.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

    model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Flatten())

    model.add(ks.layers.Dense(units=512, activation='relu'))
    model.add(ks.layers.Dropout(0.5))

    model.add(ks.layers.Dense(units=NUM_CLASSES))

    model.add(ks.layers.Softmax(name="result"))

    return model


# Preparing data for training
(x_train, y_train), (x_test, y_test) = get_data()
x_train_scaled = scale(x_train)
x_test_scaled = scale(x_test)
y_train = ks.utils.to_categorical(y_train, NUM_CLASSES)
y_test = ks.utils.to_categorical(y_test, NUM_CLASSES)

# Tensorboard data
logdir = "logs\\scalars\\" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = ks.callbacks.TensorBoard(log_dir=logdir)

# Create, compile, fit and save the model
# model = create_model()
model = ks.models.load_model("model_v1.h5")
model.compile(loss='categorical_crossentropy',
              optimizer=ks.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

model.fit(x_train_scaled, y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=2,
          shuffle=True,
          validation_data=(x_test_scaled, y_test),
          callbacks=[tensorboard_callback])

score = model.evaluate(x_test_scaled, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(SAVE_PATH)
print("Model saved at ", SAVE_PATH)

# Model v1
# Epoch 100