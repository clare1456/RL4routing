import numpy as np
import os

root_path = os.path.dirname(os.path.abspath(__file__))

def get_root_path():
    return root_path

def log_info(file_name:str, log:str, tail:str=".txt", is_append:bool=True):
    if is_append:
        with open(root_path+'/mylog/'+file_name+tail, 'a') as file:
            file.write(log+'\n')
    else:
        with open(root_path+'/mylog/'+file_name+tail, 'w') as file:
            file.write(log+'\n')

def cal_state(state_info):
    state = np.zeros(len(state_info) - 1)
    for i in range(1, len(state_info)):
        if abs(state_info[0][0] - state_info[i][0]) < 1e-6:
            if abs(state_info[0][1] - state_info[i][1]) < 1e-6:
                state[i-1] = 1
    return state

def cal_node(state):
    node = np.argmax(state)
    return node