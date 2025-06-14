import os
import pickle

def get_GO_def_set(path='../../data/source/uniport/GO_id_def.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            GO_def_set = pickle.load(file)
    else:
        with open('../../data/source/uniport/go-basic.obo') as f:
            info_list = f.readlines()

        GO_id = ""
        GO_def_set = {}

        for info in info_list:
            if "id: GO" in info:
                GO_id = info.split(" ")[1].strip()

            if "def:" in info and GO_id != "":
                GO_def = info.split("\"")[1].strip()
                GO_def_set[GO_id] = GO_def
                GO_id = ""

        with open('../../data/source/uniport/GO_id_def.pkl', "wb") as file:
            pickle.dump(GO_def_set, file)

    return GO_def_set



