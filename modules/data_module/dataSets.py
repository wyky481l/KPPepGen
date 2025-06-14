import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as pyg_Dataset
from models.KgTripleModel import get_pathogen_KG_feature_from_pretraining
from modules.sequence.encoder import SeqNoise
from util.constant import go_relation_type_dict
from util.embedding.sequence import fasta_to_index, index_to_onehot


class GoGoTripleDataSet(Dataset):
    def __init__(self):
        triple_data = pd.read_csv("data/source/uniport/go_go_triple.csv")

        self.head_list = list(triple_data.heads)
        self.relation_list = list(triple_data.relations)
        self.tail_list = list(triple_data.tails)

        self.go_term_list = list(set(self.tail_list))
        self.go_sampler = torch.distributions.Categorical(torch.ones(len(self.go_term_list)))

        self.num_go_relation_classes = len(list(go_relation_type_dict))
        self.head_type = 0

    def __getitem__(self, index):
        head_go_id = str(self.head_list[index])
        tail_go_id = str(self.tail_list[index])
        relation = self.relation_list[index]

        relation_feature = index_to_onehot([go_relation_type_dict[relation]], self.num_go_relation_classes)

        neg_go_id = str(self.go_term_list[self.go_sampler.sample()])

        return head_go_id, relation_feature, tail_go_id, neg_go_id, self.head_type

    def __len__(self):
        return len(self.head_list)


class PeptideGoTripleDataSet(Dataset):
    def __init__(self):
        triple_data = pd.read_csv("data/source/uniport/peptide_go_triple.csv")
        # loading triple data

        self.head_list = list(triple_data.pdb_id)
        self.tail_list = list(triple_data.go_info)

        self.go_term_list = list(set(self.tail_list))
        self.go_sampler = torch.distributions.Categorical(torch.ones(len(self.go_term_list)))
        # Construct sampler for go_term, where each term is uniformly sampled

        self.num_go_relation_classes = len(list(go_relation_type_dict))
        self.head_type = 1
        # head_type=1 -> peptide-go triples

    def __getitem__(self, index):
        sequence_id = str(self.head_list[index])
        go_term_id = str(self.tail_list[index])
        relation = "peptide_go"

        relation_feature = index_to_onehot([go_relation_type_dict[relation]], self.num_go_relation_classes)

        neg_go_id = str(self.go_term_list[self.go_sampler.sample()])

        return sequence_id, relation_feature, go_term_id, neg_go_id, self.head_type

    def __len__(self):
        return len(self.head_list)


class PepPathogenTripleDataSet(Dataset):
    def __init__(self):
        triple_data = pd.read_csv("data/source/amp/peptide_pathogen_triple.csv")
        self.head_list = list(triple_data.seq_id)
        self.tail_list = list(triple_data.pathogen)

        self.pathogen_list = list(set(self.tail_list))
        self.pathogen_sampler = torch.distributions.Categorical(torch.ones(len(self.pathogen_list)))

    def __getitem__(self, index):
        head_seq_data = self.head_list[index]
        tail_pathogen_data = self.tail_list[index]
        neg_tail_pathogen_data = self.pathogen_list[self.pathogen_sampler.sample()]

        return head_seq_data, tail_pathogen_data, neg_tail_pathogen_data

    def __len__(self):
        return len(self.head_list)


class PepSequenceDataSet(pyg_Dataset):
    def __init__(self, fasta_list):
        super().__init__()
        self.fasta_list = fasta_list

    def __getitem__(self, index):
        seq = self.fasta_list[index]
        seq_logit = index_to_onehot(fasta_to_index(seq))

        sequence = Data(
            x=seq_logit,
            seq=seq,
            length=len(seq)
        )
        return sequence

    def __len__(self):
        return len(self.fasta_list)


class PepSequencePathogenDataSet(pyg_Dataset):
    def __init__(self):
        super().__init__()

        data = pd.read_csv('data/source/amp/peptide_pathogen_triple.csv')
        self.fasta_list = data['sequence'].tolist()
        self.pathogen_list = data['pathogen'].tolist()

        self.pathogen_kg_feature = get_pathogen_KG_feature_from_pretraining()
        self.prior_noise = SeqNoise()

    def __getitem__(self, index):
        seq = self.fasta_list[index]
        pathogen_type = self.pathogen_list[index]

        seq_logit = index_to_onehot(fasta_to_index(seq))
        pathogen_kg_feature = self.pathogen_kg_feature[pathogen_type]

        if self.pathogen_prior:
            pathogen_noise = self.prior_noise.get_noise(pathogen_type=pathogen_type)
        else:
            pathogen_noise = self.prior_noise.get_noise(pathogen_type=None)

        pathogen_data = Data(
            x=seq_logit,
            seq=seq,
            length=len(seq),
            kg_feature=pathogen_kg_feature,
            noise=pathogen_noise
        )

        return pathogen_data

    def __len__(self):
        return len(self.fasta_list)


class PepSequenceConditionDataSet(pyg_Dataset):
    def __init__(self):
        super().__init__()

        data = pd.read_csv('data/source/amp/peptide_pathogen_triple.csv')
        self.fasta_list = data['sequence'].tolist()
        self.pathogen_list = data['pathogen'].tolist()

        self.pathogen_kg_feature = get_pathogen_KG_feature_from_pretraining()

        feature_list = torch.cat(list(self.pathogen_kg_feature.values()), dim=0)
        self.normal_feature = torch.mean(feature_list, dim=0).unsqueeze(0)

        self.prior_noise = SeqNoise()

    def __getitem__(self, index):
        seq = self.fasta_list[index]
        pathogen_type = self.pathogen_list[index]

        seq_logit = index_to_onehot(fasta_to_index(seq))
        pathogen_kg_feature = self.pathogen_kg_feature[pathogen_type]

        condition_noise = self.prior_noise.get_noise(pathogen_type=pathogen_type)
        normal_noise = self.prior_noise.get_noise(pathogen_type=None)

        pathogen_data = Data(
            x=seq_logit,
            seq=seq,
            length=len(seq),
            condition_feature=pathogen_kg_feature,
            normal_feature=self.normal_feature,
            condition_noise=condition_noise,
            normal_noise=normal_noise
        )

        return pathogen_data

    def __len__(self):
        return len(self.fasta_list)
