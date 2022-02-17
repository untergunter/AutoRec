import pickle
import torch
import os

def save_obj(obj, name ):
    with open('obj/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_model(model_class, trainer, activation, hidden_size, lr, λ, is_default_dataset):
    path = f"obj/{model_class.__name__}_{activation.__name__}_{hidden_size}_{lr}_{λ}_{is_default_dataset}_model_dict.ckpt"
    trainer.save_checkpoint(path)
    return path


def load_model(model_class, activation, hidden_size, lr, λ, is_default_dataset):
    path = f"obj/{model_class.__name__}_{activation.__name__}_{hidden_size}_{lr}_{λ}_{is_default_dataset}_model_dict.ckpt"
    model = model_class.load_from_checkpoint(path)
    model.eval()
    return model

def get_model_attr_from_path(path):
    class_name,activation,hidden_size,lr,λ,data_set_name = path.split(os.sep)[-1].replace('_model_dict.ckpt','').split('_')
    return class_name,activation,hidden_size,lr,λ,data_set_name