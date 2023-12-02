import numpy as np
import os

root_path = os.path.dirname(os.path.abspath(__file__))
file_set = set()

def get_root_path():
    return root_path

def log_info(file_name:str, log:str, tail:str=".txt", is_append:bool=True):
    file_full_name = file_name + tail
    if not file_set.__contains__(file_full_name):
        is_append = False
        file_set.add(file_full_name)
    if is_append:
        with open(root_path+'/mylog/'+file_full_name, 'a') as file:
            file.write(log+'\n')
    else:
        with open(root_path+'/mylog/'+file_full_name, 'w') as file:
            file.write(log+'\n')
        