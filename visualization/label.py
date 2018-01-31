"""
These functions are for visualizing metrics concerning a single label
"""
import matplotlib.pyplot as plt
from metrics.common_metrics import accuracy,precision,recall

def label_precision(train_label_confusion,val_label_confusion,label,save_path=None):
    """
    Plot precision per epoch
    :param train_label_confusion: the confusion matrix (as a tuple) for a single label during training time
    :param val_label_confusion: the confusion matrix (as a tuple) for a single label during validation time
    :param label: label name (for determining plot title)
    :param save_path:
    :return:
    """
    train_precision = precision(*train_label_confusion)
    val_precision = precision(*val_label_confusion)
    plt.plot(range(len(train_precision)), train_precision, label='train')
    plt.plot(range(len(val_precision)), val_precision, label='val')
    plt.title('Precision for label {}'.format(label))
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def label_recall(train_label_confusion, val_label_confusion, label, save_path=None):
    """
    Plot recall per epoch
    :param train_label_confusion: the confusion matrix (as a tuple) for a single label during training time
    :param val_label_confusion: the confusion matrix (as a tuple) for a single label during validation time
    :param label: label name (for determining plot title)
    :param save_path:
    :return:
    """
    train_recall = recall(*train_label_confusion)
    val_recall = recall(*val_label_confusion)
    plt.plot(range(len(train_recall)), train_recall, label='train')
    plt.plot(range(len(val_recall)), val_recall, label='val')
    plt.title('Recall for label {}'.format(label))
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

def label_precision_recall(train_label_confusion,val_label_confusion,label,save_path=None):
    """
    Plot the precision-recall curve for a single label.
    :param train_label_confusion: the confusion matrix (as a tuple) for a single label during training time
    :param val_label_confusion: the confusion matrix (as a tuple) for a single label during validation time
    :param label: label name (for determining plot title)
    :return:
    """
    # Calculate precision and recall for train and val
    train_precision = precision(*train_label_confusion)
    train_recall = recall(*train_label_confusion)
    val_precision = precision(*val_label_confusion)
    val_recall = recall(*val_label_confusion)

    # Plot the precision-recall curve for both train and val
    plt.plot(train_precision, train_recall, label='train')
    plt.plot(val_precision, val_recall, label='val')
    plt.title('Precision-Recall for label {}'.format(label))
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

def label_accuracy(train_label_confusion,val_label_confusion,label,save_path=None):
    """
    Plot the overall accuracy for a single label based on the label confusion.
    NOTE: This plot may be misleading, not recommended unless you have a darned good reason to use this.
    :param train_label_confusion: the confusion matrix (as a tuple) for a single label during training time
    :param val_label_confusion: the confusion matrix (as a tuple) for a single label during validation time
    :param label: label name (for determining plot title)
    :param save_path:
    :return:
    """
    train_acc = accuracy(*train_label_confusion)
    val_acc = accuracy(*val_label_confusion)

    # Plot the precision-recall curve for both train and val
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.title('Accuracy per epoch for label {}'.format(label))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()