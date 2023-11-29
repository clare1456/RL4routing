import numpy as np
import random
import torch.utils.tensorboard as tb

class Writer:
    def __init__(self, args):
        self.args = args
        if self.args.log_flag:
            # build writer
            self.writer = tb.SummaryWriter(log_dir=self.args.log_path)
            self.value_record_dict = {}
            self.step_record_dict = {}
            # record args
            self.writer.add_text("args", str(self.args.__dict__)) 
    
    def add_scalar(self, tag, value, step):
        # add scalar to tensorboard
        if not self.args.log_flag:
            return
        self.writer.add_scalar(tag, value, step)

    def add_scalar_to_buffer(self, tag, value, step):
        # add scalar to buffer
        if not self.args.log_flag:
            return
        if tag in self.value_record_dict:
            self.value_record_dict[tag].append(value)
        else:
            self.value_record_dict[tag] = [value]
        self.step_record_dict[tag] = step
    
    def buffer_update(self):
        # add scalar from buffer to tensorboard
        if not self.args.log_flag:
            return
        for tag, values in self.value_record_dict.items():
            self.add_scalar(tag, np.mean(values), self.step_record_dict[tag])
            self.value_record_dict[tag] = []

