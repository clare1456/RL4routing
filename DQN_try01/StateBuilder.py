import numpy as np
import torch

class StateBuilder():
    def __init__(self) -> None:
        pass   

    def get_state_dim(self):        
        """
        return the state space dimention
        """
        raise NotImplementedError

    def cal_state(self, state_info, info):
        """
        cal state with the infomation get from env
        """
        raise NotImplementedError

    def cal_node(self, state, state_info, info):
        """
        cal node id with state
        """
        raise NotImplementedError
    

class StateBuilder_node01(StateBuilder):
    def __init__(self, node_num:int) -> None:
        super(StateBuilder_node01, self).__init__()
        self.node_num = node_num
        self.state_dim = node_num
        pass   

    def get_state_dim(self):        
        return self.state_dim

    def cal_state(self, state_info, info):
        state = np.zeros(self.state_dim)
        find_flag = False
        for i in range(2, len(state_info)):
            if state_info[0][0] == state_info[i][0] and state_info[0][1] == state_info[i][1]:
                find_flag = True
                state[i-2] += 1
        if not find_flag:
            print("[zy] ERROR: StateBuilder_node01_mask.cal_state() cannot find state")
        return state

    def cal_node(self, state, state_info, info):
        node = np.argmax(state)
        return node

class StateBuilder_node01_mask(StateBuilder):
    def __init__(self, node_num:int) -> None:
        super(StateBuilder_node01_mask, self).__init__()
        self.node_num = node_num
        self.state_dim = node_num + node_num
        pass   

    def get_state_dim(self):        
        return self.state_dim

    def cal_state(self, state_info, info):
        state = np.zeros(self.state_dim)
        find_flag = False
        for i in range(2, len(state_info)):
            if state_info[0][0] == state_info[i][0] and state_info[0][1] == state_info[i][1]:
                find_flag = True
                state[i-2] += 1
        if not find_flag:
            print("[zy] ERROR: StateBuilder_node01_mask.cal_state() cannot find state")
        for i in range(self.node_num):
            if info["mask"][i]:
                state[self.node_num + i] += 1
        return state

    def cal_node(self, state, state_info, info):
        node = np.argmax(state[:self.node_num])
        return node

class StateBuilder_node_now_xy(StateBuilder):
    def __init__(self, node_num:int) -> None:
        super(StateBuilder_node_now_xy, self).__init__()
        self.node_num = node_num
        self.state_dim = 2
        pass   

    def get_state_dim(self):        
        return self.state_dim

    def cal_state(self, state_info, info):
        state = np.array(state_info[0][:2])
        return state

    def cal_node(self, state, state_info, info):
        for i in range(self.node_num):
            if state[0] == state_info[2+i][0] and state[1] == state_info[2+i][1]:
                return i
        print("[zy] ERROR: StateBuilder_node_now_xy.cal_node() cannot find node")
        return -1

class StateBuilder_node_all_xy(StateBuilder):
    def __init__(self, node_num:int) -> None:
        super(StateBuilder_node_all_xy, self).__init__()
        self.node_num = node_num
        self.state_dim = 2 * (node_num + 2)
        pass   

    def get_state_dim(self):        
        return self.state_dim

    def cal_state(self, state_info, info):
        state = torch.tensor(state_info[:, :2])
        state = state.view(-1)
        return state

    def cal_node(self, state, state_info, info):
        for i in range(self.node_num):
            if state[0] == state_info[2+i][0] and state[1] == state_info[2+i][1]:
                return i
        print("[zy] ERROR: StateBuilder_node_all_xy.cal_node() cannot find node")
        return -1

class StateBuilder_node01_mask_all_xy(StateBuilder):
    def __init__(self, node_num:int) -> None:
        super(StateBuilder_node01_mask_all_xy, self).__init__()        
        self.node_num = node_num
        self.sub_builder_1 = StateBuilder_node01_mask(node_num)
        self.sub_builder_2 = StateBuilder_node_all_xy(node_num)
        self.state_dim = self.sub_builder_1.state_dim + self.sub_builder_2.state_dim
        pass   

    def get_state_dim(self):        
        return self.state_dim

    def cal_state(self, state_info, info):
        state = np.zeros(self.state_dim)
        state[:self.sub_builder_1.state_dim] = self.sub_builder_1.cal_state(state_info, info)
        state[self.sub_builder_1.state_dim:] = self.sub_builder_2.cal_state(state_info, info)
        return state

    def cal_node(self, state, state_info, info):
        node = np.argmax(state[:self.node_num])
        return node