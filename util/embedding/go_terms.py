import os
import pickle
import numpy as np
import pandas as pd
import torch
from goatools.obo_parser import GODag
from paddlenlp.transformers import BertModel, AutoTokenizer
from tqdm import tqdm


def get_go_prefeature(path="data/source/uniport/go_prefeature.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result

    model = BertModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    MAX_TEXT_SEQ_LENGTH = 128

    with open("data/source/uniport/GO_id_def.pkl", "rb") as f:
        data = pickle.load(f)

    result = {}
    for go_key in tqdm(list(data.keys())):
        text = data[go_key]

        encoded_input = tokenizer(text, max_length=MAX_TEXT_SEQ_LENGTH, padding='max_length', return_tensors='pd')

        with torch.no_grad():
            output = model(**encoded_input)

        result[go_key] = [text, output[1].numpy()]

    with open(path, "wb") as f:
        pickle.dump(result, f)

    return result


def get_go_go_triple():
    path = "data/source/uniport/go_go_triple.csv"
    if os.path.exists(path):
        triple_data = pd.read_csv(path)
        return triple_data

    godag = GODag('data/source/uniport/go-basic.obo', optional_attrs={'relationship'})
    heads = []
    relations = []
    tails = []

    for go_id in tqdm(godag.keys()):
        is_a_terms = list(godag[go_id].parents)
        if not is_a_terms:
            continue

        for term in is_a_terms:
            heads.append(go_id)
            relations.append("is_a")
            tails.append(term.id)

        relationship = godag[go_id].relationship
        if not relationship:
            continue

        relation_list = ["part_of", "regulates", "negatively_regulates", "positively_regulates"]
        for relation in relation_list:
            if relation in relationship:
                for term in list(relationship[relation]):
                    heads.append(go_id)
                    relations.append(relation)
                    tails.append(term.id)

    triple_data = pd.DataFrame({"heads": heads, "relations": relations, "tails": tails})
    triple_data.to_csv('data/source/uniport/go_go_triple.csv')

    return triple_data


def get_go_feature_from_term_list(id_list, go_prefeature_dict, device):
    feature_list = [go_prefeature_dict[term_id][1] for term_id in id_list]
    result = torch.tensor(np.stack(feature_list), device=device)
    result = result.squeeze(dim=1)
    return result
