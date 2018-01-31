"""
These functions are related directly to the confusion matrix
"""

import matplotlib.pyplot as plt

def plot_cell_per_epoch(train_cf,val_cf,cell_idx):
    train_cell = train_cf[:,cell_idx[0],cell_idx[1]]
    val_cell = val_cf[:,cell_idx[0],cell_idx[1]]
    plt.plot(train_cell)
    plt.plot(val_cell)
    plt.show()

def plot_matrix(matrix,min_val,max_val):
    im = plt.imshow(matrix,cmap='RdYlGn',vmin=min_val,vmax=max_val)
    plt.colorbar(im)
    plt.show()