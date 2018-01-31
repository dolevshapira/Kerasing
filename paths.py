import os
from pathlib import Path
src_dir = Path(os.getcwd())
data_dir = src_dir.joinpath('data')
data_dir.mkdir(exist_ok=True,parents=True)
# All kinds of data directories according to the workflow
# Input related files
input_dir = data_dir.joinpath('input')
input_dir.mkdir(exist_ok=True,parents=True)

preprocessed_samples_train = str(input_dir.joinpath('preprocessed_train_samples.h5'))
preprocessed_targets_train = str(input_dir.joinpath('preprocessed_train_targets.h5'))
preprocessed_samples_test = str(input_dir.joinpath('preprocessed_test_samples.h5'))
preprocessed_targets_test = str(input_dir.joinpath('preprocessed_test_targets.h5'))

normalized_train_samples =str(input_dir.joinpath('normalized_train_samples.h5'))
normalized_train_targets =str(input_dir.joinpath('normalized_train_targets.h5'))
normalized_test_samples = str(input_dir.joinpath('normalized_test_samples.h5'))
normalized_test_targets = str(input_dir.joinpath('normalized_test_targets.h5'))

# Models related files
saved_models_dir = data_dir.joinpath('saved models')
saved_models_dir.mkdir(exist_ok=True,parents=True)

model_file_format = str(saved_models_dir.joinpath('Model on epoch {epoch:03d}.h5'))
train_cf_matrix = str(saved_models_dir.joinpath('train_cf_matrix.h5'))
val_cf_matrix = str(saved_models_dir.joinpath('val_cf_matrix.h5'))

# Output related files
output_dir = data_dir.joinpath('output')
output_dir.mkdir(exist_ok=True,parents=True)