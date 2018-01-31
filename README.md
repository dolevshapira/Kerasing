# Keras Boilerplate
A boilerplate (sort-of) for Keras Projects.
I suggest using it as a boilerplate for Image related projects.

* preprocess.py - preprocessing logic to create a 
pre-normalized, pre-augmented, pre-regularized data. 
Simply put, this should prepare your raw data to an easy to work with format.
* normalize.py - normalization, augmentation, regularization logic. 
This process should output a format easy to feed to the model chosen for your problem
* train.py - model creation and training logic. 
It should use the data from the normalize process to generate a model and statistics for the visualize process
such as confusion matrix, accuracy, lossn etc.
* visualize.py - create plots for better understanding of the data and training process.

# Examples
The first thing I'd recommend checking.

# DAL
A common name for data-access layer, meaning all data handling objects
Use these tools to handle datasets and pre-processed results.

Use Dataset abstract class as an interface for implementing any other dataset objects
such as Oracle / MySQL databases, or perhaps Elastic Search

# Callbacks
Add custom callbacks to this package.

# Metrics
Add custom metrics to this package.