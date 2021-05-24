from data.dataset import *
from pandas import read_csv
import numpy as np


#hyperparameters
dataset_path = r"sets/income.csv"
batch_size = 498
validate_every_no_of_batches = 498
epochs = 10
input_size = 1
output_size = 1
hidden_shapes = [0]
loss = 'mse'
lr = 0.0085
has_dropout=True
dropout_perc=0.5
output_log = r"runs/income.txt"

with open(dataset_path, "rb") as input_file:
    income_dataset = read_csv(input_file)
    x = np.array(income_dataset['income']).reshape(-1, 1)
    x = x / np.max(x, axis=0)
    y = np.array(income_dataset['happiness'])
data = dataset(x, y, batch_size)
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.8, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
