from dal.hdf5dataset import Hdf5Dataset
import paths as p
from hyper_parameters import num_classes
from keras.utils import to_categorical
import numpy as np

def normalize_sample(sample):
    return sample / 255

def normalize_target(target,num_classes):
    return to_categorical(target,num_classes=num_classes)

def normalize_ds(samples,targets,normalized_samples,normalized_targets):
    print('Shuffling')

    # You can't shuffle when fitting during training, so one may say shuffling is regularization / normalization
    order = np.arange(samples.shape[0])
    np.random.shuffle(order)
    print('Normalizing data')

    # Normalize and save each entry from the samples in a shuffled order
    for idx in order:
        normalized_samples.insert(normalize_sample(samples[idx]))
        normalized_targets.insert(normalize_target(targets[idx], num_classes))

if __name__ == '__main__':
    print('Creating Normalized Training Datasets')

    # Normalize training data
    print('Normalizing training data')
    print('Opening training datasets')
    with Hdf5Dataset(p.preprocessed_samples_train) as samples, Hdf5Dataset(p.preprocessed_targets_train) as targets, \
        Hdf5Dataset(p.normalized_train_samples,shape=samples.shape[1:]) as normalized_samples, \
        Hdf5Dataset(p.normalized_train_targets, shape=(num_classes,)) as normalized_targets:
        normalize_ds(samples,targets,normalized_samples,normalized_targets)
        print('Closing training datasets')

    # Normalizing test data
    print('Normalizing test data')
    print('Opening test datasets')
    with Hdf5Dataset(p.preprocessed_samples_test) as samples, Hdf5Dataset(p.preprocessed_targets_test) as targets, \
        Hdf5Dataset(p.normalized_test_samples,shape=samples.shape[1:]) as normalized_samples, \
        Hdf5Dataset(p.normalized_test_targets, shape=(num_classes,)) as normalized_targets:
        normalize_ds(samples, targets, normalized_samples, normalized_targets)
        print('Closing test datasets')

    print('Normalization Finished')