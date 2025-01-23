from torch.utils.data import Dataset
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os


class TextDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        return self.data_list[index]
    

def set_seed(seed=42):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
    np.random.seed(seed)
    random.seed(seed)


def draw_plot(x, y, data_path):
    
    plt.plot(x, y, marker='o', color='r')
    plt.title('Variation of accuracy with epoch')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.savefig(data_path)


def get_date():
 
    current_time = datetime.now()
    ymd = current_time.strftime("%Y%m%d")
    hms = current_time.strftime("%H%M%S")
    return ymd, hms

def makedir(path):

    for path in path.values():
        if not os.path.exists(path):
            os.makedirs(path)
