import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers.experimental import preprocessing
#import time
from keras.layers import Dropout
import pandas as pd
import sys


def load_model(parent_dir):
    path = os.path.join(parent_dir, 'packet_generator2')
    return tf.saved_model.load(path)


def create_tabular_data(dataset, column_names, tcp_flags_index = 20, file_name = 'Model_packet_data_output', save=False):
  print('Creating tabular data...')
  # Create a pandas dataframe from the processed dataset:
  dataframe = pd.DataFrame(dataset, columns= column_names)
  # Restore to natural order of columns to match with the original dataset:
  column  = column_names.pop(0)
  # Take the column name 'tcp.flags' to its original position in the dataset:
  column_names.insert(tcp_flags_index,column)
  # Restore
  dataframe = dataframe[column_names]
  # Decode all strings in each column from bytes to 'utf-8':
  for col in column_names:
    dataframe[col] = dataframe[col].str.decode('utf-8')
  # If save , save to the provided file.
  print(file_name)
  if save:
    dataframe.to_csv(file_name)
    print(f'Dataset saved to file named : {file_name}')
    return 
  else:
    return 

def process_data(dataset):
  # Replace all occurences of '>', '<', '<p>' with ''.
  dataset = tf.strings.regex_replace(dataset, b'<p>', b'')
  dataset = tf.strings.regex_replace(dataset, b'>', b'')
  dataset = tf.strings.regex_replace(dataset, b'<', b'')
  # In the innermost axis, join all the elements of the list together to form one long string:
  dataset = tf.strings.reduce_join(dataset, axis = -1)
  
  
  return dataset.numpy()

# Function that loads column names from memory:
def get_column_names():
  with open('columns.pkl','rb') as f:
    columns = pickle.load(f)
  return columns

# Function that load the column datatypes from storage:
def get_columns_dtype():
  with open('column_dtypes.pkl', 'rb') as f:
    columns_dtype = pickle.load(f)
  return columns_dtype

def generate(data_size, column_names=None, batch_size=2000, temperature=1.0, dir_path= None, save=False, file_name='packet_data_output.csv'):
  '''
    Function generates packet data.
    Args:
    data_size : (int) number of samples to be generated.
    column_names: (list) list of the column names in the dataset (e.g 'tcp.flags', 'ip.id')
    batch_size: (int) size of each batch for generation per time step.
    temperature: (float) controls the degree of randomness of model output during generation.
    dir_path: (string) parent directory that directly contains the saved model and all its dependencies.
    save : (bool) whether to save the dataset generated or not.
    file_name: (string) name of the csv file used to save the generated dataset.
  ''' 
  
  print('Loading model to memory...')
  # Load model to memory:
  # If parent directory not provided, use the current working directory
  if dir_path :
    try:
      model  = load_model(dir_path)
    else:
      print('Model not found in the directory provided. Please try again.')
      return
  else:
    dir_path = os.path.abspath(os.curdir)
    try:
      model = load_model(dir_path)
    except:
      print('Model not found in the current working directory. Please try again.')
      return 
    
  # If batch_size is not 100, 500, 1000, 0r 2000, print the below:
  if batch_size not in [100, 500, 1000, 2000]:
      print('Model only takes batch sizes of : 100, 500, 1000, 2000. Please, use one of these predefined sizes.')
      return 
  
  print('Generating packet data ...')
  # If data_size < batch_size, make data_size the batch_size:
  if data_size < batch_size:
    try:
      data = model.generate_samples(batch_size=data_size, temperature =temperature)
    except:
      print('Model only takes batch sizes of : 100, 500, 1000, 2000. Please, use one of these predefined sizes or a size larger and that is a multiple of the batch size used.')
      return
  else:
    # Else use the batch_size to generate subsequent batches of generated datasets till the data_size is met:
    incorrect_size = True
    data = None
    while (incorrect_size):
      data = model.generate_samples(batch_size=batch_size, temperature =temperature)
      if data.shape[-1] < 14:
        continue
      else:
        incorrect_size = False
    for i in range(batch_size, data_size, batch_size):
      unequal_size = True
      while (unequal_size):
        temp_data = model.generate_samples(batch_size=batch_size, temperature=temperature)
        if temp_data.shape[-1] < 14:
          continue
        else:
          data  = tf.concat([data, temp_data], axis=0)
          unequal_size = False

  # process_data: remove '>', '<', '<p>' strings from all items:
  data = process_data(data)
  if column_names == None:
    column_names = get_column_names()
    create_tabular_data(data, column_names=column_names, file_name=file_name, save=save)
  else:
    create_tabular_data(data, column_names=column_names, file_name=file_name, save=save)
  return 



if __name__ == '__main__':
    # Call file like this in command line: 'Packet_generator.py data_size(data type :int) file_name(data type : string) dir_path(data type : string)'
    # data_size: number of samples to be generated.
    # file_name: name of the file to save generated dataset
    # dir_path: (optional) parent directory that hosts the trained model and all its dependencies (e.g columns.pkl,  columns_dtype.pkl files). If not specified, the program will use the current working directory.
    if sys.argv[3]:
      dir_path = sys.argv[3]
    else:
      dir_path = os.path.abspath(os.curdir)
      
    generate(sys.argv[1], dir_path =dir_path , save=True, file_name=sys.argv[2])
    