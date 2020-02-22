import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import datetime as dt

SAVE_PATH = "model_datagen_v1.h5"
NUM_CLASSES = 10
SHAPE = (32, 32, 3)
EPOCHS = 100
BATCH_SIZE = 64


# load train and test dataset
def get_data():
    # load dataset
    (train_x, train_y), (test_x, test_y) = ks.datasets.cifar10.load_data()

    # one hot encoding
    train_y = ks.utils.to_categorical(train_y, NUM_CLASSES)
    test_y = ks.utils.to_categorical(test_y, NUM_CLASSES)

    return train_x, train_y, test_x, test_y


# scale images
def scale(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm


# define cnn model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=SHAPE, name="input_image"))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(NUM_CLASSES, activation='softmax', name="result"))

    # compile model
    model.compile(optimizer=ks.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# run the fitting and evaluation
def run():
    # load dataset
    train_x, train_y, test_x, test_y = get_data()

    # scaling images
    train_x, test_x = scale(train_x, test_x)

    # Tensorboard data callback
    logdir = "logs\\scalars\\" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = ks.callbacks.TensorBoard(log_dir=logdir)

    # Checkpoint callback
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint_callback = ks.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

    # define model
    model = create_model()

    # create data generator
    datagen = ks.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                        height_shift_range=0.1,
                                                        horizontal_flip=True)
    # prepare iterator
    it_train = datagen.flow(train_x, train_y, batch_size=BATCH_SIZE)

    # fit model
    steps = int(train_x.shape[0] / 64)
    model.fit_generator(it_train,
                        steps_per_epoch=steps,
                        epochs=EPOCHS,
                        validation_data=(test_x, test_y),
                        verbose=1,
                        callbacks=[tensorboard_callback, checkpoint_callback])

    # evaluate model
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(SAVE_PATH)
    print("Model saved at ", SAVE_PATH)


if __name__ == "__main__":
    run()


# Model data generator v1
# Test loss: 0.6367
# Test accuracy: 0.8337
