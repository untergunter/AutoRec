import torch
import torch.nn as nn
import pytorch_lightning as pl
from models import AutoRecBase


class VarAutoRec(AutoRecBase):
    def __init__(self,
                 number_of_items: int,
                 hidden_size: int,
                 activation_function_1,
                 activation_function_2,
                 loss):
        super(VarAutoRec, self).__init__(number_of_items=number_of_items,
                                         hidden_size=hidden_size,
                                         activation_function_1=activation_function_1,
                                         activation_function_2=activation_function_2,
                                         loss=loss)

