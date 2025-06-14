import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from paddlenlp.transformers import BertModel, AutoTokenizer
from tqdm import tqdm


def get_pathogen_prefeature():
    path = "data/source/amp/pathogen_prefeature.pkl"

    if os.path.exists(path):
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result

    model = BertModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    MAX_TEXT_SEQ_LENGTH = 128
    data = pd.read_csv("data/source/amp/pathogen_description.csv")

    result = {}
    for index in tqdm(range(len(data))):
        pathogen = data.pathogen[index]
        types = data.types[index]
        description = data.description[index]

        encoded_input = tokenizer(description, max_length=MAX_TEXT_SEQ_LENGTH, padding='max_length',
                                  return_tensors='pd')

        with torch.no_grad():
            output = model(**encoded_input)

        result[pathogen] = [types, description, output[1].numpy()]

    with open(path, "wb") as f:
        pickle.dump(result, f)

    return result


def get_pathogen_triple():
    pathogen_triple = pd.read_csv("data/source/amp/pathogen_description.csv")

    head_list = list(pathogen_triple.pathogen[:-3])
    tail_list = list(pathogen_triple.types[:-3])

    nag_pathogen = set(list(pathogen_triple.pathogen[-3:]))

    neg_tail_list = []
    for tail in tail_list:
        data = list(nag_pathogen.difference({tail}))
        neg_tail_list.append(data[random.randint(0, 1)])

    return head_list, tail_list, neg_tail_list


def get_pathogen_feature_from_term_list(id_list, pathogen_prefeature_dict, device):
    feature_list = [pathogen_prefeature_dict[term_id][2] for term_id in id_list]
    result = torch.tensor(np.stack(feature_list), device=device)
    result = result.squeeze(dim=1)
    return result
