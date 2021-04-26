import torch
import pickle
import numpy as np

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def read_keys_pickle(path):
    data = []
    with open(path, "rb") as fread:
        reader = pickle.Unpickler(fread)
        while True:
            try:
                data.append(reader.load())
            except EOFError:
                break
    return data

def save_keys_pickle(data, path):
    with open(path, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        for line in data:
            pickler.dump(line)