import pickle
import torch


def save_obj(obj, name ):
    with open('obj/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_model(model, trainer, activation, hidden_size, lr, λ, is_default_dataset):
    path = f"obj/{type(model).__name__}_{activation}_{hidden_size}_{lr}_{λ}_{is_default_dataset}_model_dict.ckpt"
    trainer.save_checkpoint(path)


def load_model(model, activation, hidden_size, lr, λ, is_default_dataset):
    path = f"obj/{type(model).__name__}_{activation}_{hidden_size}_{lr}_{λ}_{is_default_dataset}_model_dict.ckpt"
    model = model.load_from_checkpoint(path)
    model.eval()
    return model