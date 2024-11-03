import pickle

def pickle_obj(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
