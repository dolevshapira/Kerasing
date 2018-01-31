import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from callbacks.confusion_matrix import ConfusionMatrix
from dal.hdf5dataset import Hdf5Dataset
from hyper_parameters import batch_size,num_classes,epochs
import paths as p

def create_model():
    """
    Used to create the model
    :return: Keras Model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def generator(features, labels, size):
    """
    Creates a generator returning a batch of samples and targets
    credit to @grovina from stack overflow for this.
    :param features: samples
    :param labels: targets
    :param size: batch size
    :return:
    """
    while True:
        start, end = 0, size
        while end < len(features):
            s = slice(start, end)
            yield features[s], labels[s]
            start, end = end, end + size

if __name__ == '__main__':
    # Model Creation
    print('Creating model')
    model = create_model()

    print('Opening datasets')
    with Hdf5Dataset(p.normalized_train_samples) as train_samples,\
        Hdf5Dataset(p.normalized_train_targets) as train_targets, \
        Hdf5Dataset(p.normalized_test_samples) as test_samples, \
        Hdf5Dataset(p.normalized_test_targets) as test_targets, \
        Hdf5Dataset(p.train_cf_matrix, (num_classes, num_classes)) as train_cf_mat, \
        Hdf5Dataset(p.val_cf_matrix, shape=(num_classes, num_classes)) as val_cf_mat:

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model_saver = ModelCheckpoint(str(p.model_file_format))
        cf = ConfusionMatrix(10,train_cf_mat.insert,val_cf_mat.insert)
        metrics = cf.generate_metrics()

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'] + metrics)

        model.fit_generator(
            generator(train_samples, train_targets, batch_size),
            steps_per_epoch=len(train_samples) // batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=generator(test_samples, test_targets, batch_size),
            validation_steps=len(test_samples) // batch_size,
            callbacks=[cf,model_saver],
            shuffle=False)
        print('Closing datasets')
    print('Finished training')