from data.dataset import *
from pandas import read_csv
import numpy as np


#hyperparameters
dataset_path = r"sets/income.csv"
batch_size = 10
validate_every_no_of_batches = 80
epochs = 1000
input_size = 1
output_size = 1
hidden_shapes = [1]
loss = 'mse'
lr = 0.0085
has_dropout=True
dropout_perc=0.5
output_log = r"runs/income.txt"
#iris dataset
with open(dataset_path, "rb") as input_file:
    income_dataset = read_csv(input_file)
    x = np.array(income_dataset['income'])
    x = x / np.max(x, axis=0)
    y = np.array(income_dataset['happiness'])
data = dataset(x, y, batch_size)
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.8, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
