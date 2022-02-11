import pickle
import torch


def save_obj(obj, name ):
    with open('obj/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_model(model, lr):
    torch.save(model.state_dict(), f"obj/{type(model).__name__}_{lr}_model_dict.pt")


def load_model(model, lr):
    model.load_state_dict(torch.load(f"obj/{type(model).__name__}_{lr}_model_dict.pt"))
