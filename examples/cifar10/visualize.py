from paths import val_cf_matrix,train_cf_matrix
from dal.hdf5dataset import Hdf5Dataset
from metrics.common_metrics import calculate_label_confusion
from visualization.label import label_precision_recall, label_accuracy,label_precision,label_recall
from visualization.confusion_matrix import plot_cell_per_epoch,plot_matrix
import numpy as np
import paths as p

def total_accuracy(confusion_matrix):
    confusion_matrix.trace(axis1=-2,axis2=-1)/confusion_matrix.sum(axis=(-2,-1))

if __name__ == '__main__':
    with Hdf5Dataset(train_cf_matrix) as train_matrix, \
        Hdf5Dataset(val_cf_matrix) as val_matrix:
        print(train_matrix.shape)
        print(val_matrix.shape)
        for label in range(10):
            train_label_confusion = calculate_label_confusion(train_matrix, label)
            val_label_confusion = calculate_label_confusion(val_matrix, label)
            label_precision(train_label_confusion,val_label_confusion,label)
            label_recall(train_label_confusion, val_label_confusion, label)
            label_precision_recall(train_label_confusion,val_label_confusion,label)
            label_accuracy(train_label_confusion,val_label_confusion,label)
        diff = train_matrix[49]-train_matrix[0]
        plot_matrix(diff,-diff.max(),diff.max())
