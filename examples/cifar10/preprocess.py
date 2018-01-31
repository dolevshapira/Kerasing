from keras.datasets import cifar10
from dal.hdf5dataset import Hdf5Dataset
from paths import preprocessed_samples_train,preprocessed_samples_test,preprocessed_targets_train,preprocessed_targets_test

if __name__ == '__main__':
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    print('Saving training samples as HDF5')
    with Hdf5Dataset(preprocessed_samples_train,shape=x_train.shape[1:]) as train_samples:
        for sample in x_train:
            train_samples.insert(sample)

    print('Saving test samples as HDF5')
    with Hdf5Dataset(preprocessed_samples_test, shape=x_test.shape[1:]) as test_samples:
        for sample in x_test:
            test_samples.insert(sample)

    print('Saving training targets as HDF5')
    with Hdf5Dataset(preprocessed_targets_train,shape=y_train.shape[1:]) as train_targets:
        for target in y_train:
            train_targets.insert(target)

    print('Saving test targets as HDF5')
    with Hdf5Dataset(preprocessed_targets_test, shape=y_test.shape[1:]) as test_targets:
        for target in y_test:
            test_targets.insert(target)

    print('Pre-Processing Finished')