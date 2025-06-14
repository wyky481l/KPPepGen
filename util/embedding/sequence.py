import os
import pickle
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from util.constant import AA_dict, AA_type, blosum62, PAM120, hydrophobicity, AA_index, AA_helix


def index_to_onehot(x, num_classes=20):
    x = torch.tensor(x)
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'

    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)

    return x_onehot.float()


AA_one_hot = index_to_onehot(range(0, 20))
aa_index = np.array(AA_index.values[:, 1:], dtype=float)
aa_helix = np.expand_dims(np.array(AA_helix), axis=0)
aa_info = np.concatenate((aa_index, aa_helix), axis=0)


def fasta_to_index(fasta):
    index_list = [AA_dict[aa] for aa in fasta]
    return index_list


def index_to_fasta(index_list):
    fasta = [AA_type[index] for index in index_list]
    fasta = "".join(fasta)
    return fasta


def logit_to_index(logit_p, random_state=False):
    if random_state:
        D = torch.distributions.Categorical(logit_p)
        token_index = D.sample()
    else:
        token_index = logit_p.argmax(dim=-1)

    return token_index


def head_list_sort(head_pep, head_go, index_list):
    result = []
    pep_index = 0
    go_index = 0
    for index in range(len(index_list)):
        if index_list[index]:
            result.append(head_pep[pep_index])
            pep_index += 1
        else:
            result.append(head_go[go_index])
            go_index += 1

    result = torch.stack(result, dim=0)

    return result


def onehot_encoding(seq):
    encoding_map = np.eye(len(AA_type))

    residues_map = {}
    for i, aa in enumerate(AA_type):
        residues_map[aa] = encoding_map[i]

    tmp_seq = [residues_map[aa] for aa in seq]
    return np.array(tmp_seq)


def get_bio_embedding_for_sequence(fasta, fasta_index, encode_type=None):
    if fasta_index is None:
        if encode_type is None:
            encode_type = ["blosum", "pam", "hydrophobicity", "aaindex"]

        embedding = []
        for aa_type in encode_type:
            embedding.append(get_embedding_form_peptide(fasta=fasta, encode_type=aa_type))

        embedding = np.concatenate(embedding, axis=1)

        return embedding
    else:
        device = fasta_index.device
        embedding = [torch.tensor(get_blosum_embedding_from_peptide(fasta), device=device).reshape(len(fasta), -1),
                     torch.tensor(get_pam_embedding_from_peptide(fasta), device=device).reshape(len(fasta), -1),
                     torch.tensor(get_hydrophobicity_embedding_from_peptide(fasta),
                                  device=device).reshape(len(fasta), -1),
                     torch.tensor(get_aaindex_embedding_from_peptide(fasta_index), device=device).reshape(len(fasta),
                                                                                                          -1)]
        embedding = torch.cat(embedding, dim=1)
        return embedding


def get_embedding_form_peptide(fasta=None, encode_type=None):
    assert encode_type in {"blosum", "pam", "hydrophobicity", "aaindex"}, "embedding type error"

    if encode_type == "blosum":
        embedding = get_blosum_embedding_from_peptide(fasta)
    elif encode_type == "pam":
        embedding = get_pam_embedding_from_peptide(fasta)
    elif encode_type == "hydrophobicity":
        embedding = get_hydrophobicity_embedding_from_peptide(fasta)
    else:
        fasta_index = fasta_to_index(fasta)
        embedding = get_aaindex_embedding_from_peptide(fasta_index)

    return np.array(embedding).reshape(len(fasta), -1)


def get_blosum_embedding_from_peptide(fasta):
    embedding = []
    for aa in fasta:
        embedding = embedding + blosum62[aa]

    return embedding


def get_pam_embedding_from_peptide(fasta):
    embedding = []
    for aa in fasta:
        embedding = embedding + PAM120[aa]

    return embedding


def get_hydrophobicity_embedding_from_peptide(fasta, scale=100):
    embedding = []
    for aa in fasta:
        embedding = embedding + [score / scale for score in hydrophobicity[aa]]

    return embedding


def get_aaindex_embedding_from_peptide(fasta_index):
    # data = torch.tensor(aa_info, device=fasta_index.device)
    embedding = aa_info[:, fasta_index].T

    return embedding


def get_peptide_fasta(peptide_type=None):
    assert peptide_type in ["uniport", "amp"], "peptide_type should be uniport or amp"

    if peptide_type == "uniport":
        peptide_path = "data/source/uniport/uniport_peptide.fasta"
    else:
        peptide_path = "data/source/amp/amp_peptide.fasta"

    record_list = list(SeqIO.parse(peptide_path, "fasta"))
    peptide_dict = {}
    for record in record_list:
        peptide_dict[record.id] = record.seq

    return peptide_dict


def get_seq_feature_from_dataset(peptide_type="uniport", bio_status=True):
    assert peptide_type in ["uniport", "amp"], "peptide_type should be uniport or amp"

    if bio_status:
        pkl_path = f"data/source/{peptide_type}/pep_feature.pkl"
    else:
        pkl_path = f"data/source/{peptide_type}/pep_feature2.pkl"
    fasta_path = f"data/source/{peptide_type}/{peptide_type}_peptide.fasta"

    if os.path.exists(pkl_path):
        print(f"loading the preprocessing seq feature from pkl_file")
        with open(pkl_path, "rb") as f:
            feature_dict = pickle.load(f)
        return feature_dict

    record_list = list(SeqIO.parse(fasta_path, "fasta"))
    feature_dict = {}

    print(f"preprocessing {peptide_type} seq feature ")

    for record in tqdm(record_list):
        seq_id = str(record.id)
        seq = str(record.seq)
        feature_dict[seq_id] = get_feature_from_sequence(seq, bio_status=bio_status)

    with open(pkl_path, "wb") as f:
        pickle.dump(feature_dict, f)

    return feature_dict


def get_seq_from_dataset(peptide_type="uniport"):
    if peptide_type == "uniport":
        path = "data/source/uniport/uniport_peptide.fasta"
    else:
        path = "data/source/amp/amp_peptide.fasta"
    print(f"loading {peptide_type} seq")

    record_list = list(SeqIO.parse(path, "fasta"))
    sequence_dict = {}
    for record in tqdm(record_list):
        sequence_dict[str(record.id)] = str(record.seq)

    return sequence_dict


def get_feature_from_batch_sequence(pep_id_list, feature_dict, max_len, device):
    feature_list = []
    seq_len_list = []
    # padding_mask = torch.ones(n_head, len(pep_id_list), max_len, max_len, device=device)
    padding_mask = torch.ones(len(pep_id_list), max_len, device=device)

    for index, pep_id in enumerate(pep_id_list):
        feature = torch.tensor(feature_dict[pep_id], device=device)
        seq_len = feature.shape[0]

        padding = torch.zeros(51 - seq_len, feature.shape[1], device=device)
        feature = torch.cat((feature, padding))
        feature_list.append(feature)

        padding_mask[index, :seq_len] = 0
        seq_len_list.append(seq_len)

    feature_list = torch.stack(feature_list)
    feature_list = feature_list[:, :max_len, :]
    seq_len_list = torch.tensor(seq_len_list, device=device)

    padding_mask = padding_mask.bool()

    return feature_list.float(), padding_mask, seq_len_list


def get_emb_from_batch_logit(seq_logit_list, batch_index, batch_size, max_len, device, random_state=False):
    global aa_info
    aa_info = torch.tensor(aa_info, device=device)

    feature_list = []
    seq_len_list = []
    padding_mask = torch.ones(batch_size, max_len, device=device)
    seq_logit_list = seq_logit_list.to(device)

    for index in range(batch_size):
        # bool_status = batch_index == index
        # seq_logit = seq_logit_list[bool_status]
        indices = torch.nonzero(torch.eq(batch_index, index))
        seq_logit = torch.index_select(seq_logit_list, 0, indices.squeeze(), )
        feature = get_feature_from_batch_logit(seq_logit=seq_logit, device=device, random_state=random_state)
        seq_len, hidden_dim = feature.shape

        padding = torch.zeros(51 - seq_len, hidden_dim, device=device)
        feature = torch.cat((feature, padding))
        feature_list.append(feature)

        padding_mask[index, :seq_len] = 0
        seq_len_list.append(seq_len)

    feature_list = torch.stack(feature_list)
    feature_list = feature_list[:, :max_len, :]
    seq_len_list = torch.tensor(seq_len_list, device=device)

    padding_mask = padding_mask.bool()

    return feature_list.float(), padding_mask, seq_len_list


def get_feature_from_sequence(fasta=None, index=None, bio_status=True):
    if fasta is None:
        assert index is not None, "sequence_embedding error"
        fasta = index_to_fasta(index)

    feature = onehot_encoding(fasta)
    if bio_status:
        feature_bio = get_bio_embedding_for_sequence(fasta, index)
        feature = np.concatenate((feature, feature_bio), axis=1)

    return feature


def get_feature_from_batch_logit(seq_logit, device=None, bio_status=True, random_state=False):
    seq_index = logit_to_index(seq_logit, random_state)
    seq_onehot = AA_one_hot.to(device).index_select(0, seq_index)

    # seq_onehot = index_to_onehot(seq_index)
    # if bio_status:
    #     seq_bio_emd = get_bio_embedding_for_sequence(index_to_fasta(seq_index))
    #     seq_bio_emd = torch.tensor(seq_bio_emd, device=device)
    #     seq_emd = torch.cat([seq_onehot, seq_bio_emd], dim=-1)
    #     return seq_emd
    # else:
    #     return seq_onehot

    fasta = index_to_fasta(seq_index)
    seq_bio_emd = get_bio_embedding_for_sequence(fasta, seq_index)
    # seq_bio_emd = torch.tensor(seq_bio_emd, device=device)
    seq_emd = torch.cat([seq_onehot, seq_bio_emd], dim=-1)
    # seq_emd = torch.cat([seq_logit, seq_bio_emd], dim=-1)

    return seq_emd


def logit_to_index(logit_p, random_state=False):
    # token_index = logit_p.argmax(dim=-1)
    if random_state:
        D = torch.distributions.Categorical(logit_p)
        token_index = D.sample()
    else:
        token_index = logit_p.argmax(dim=-1)

    return token_index


def remove_padding(seq_pred, batch_length, attn=None):
    seq_pred_list = []
    if attn is None:
        for index, length in enumerate(batch_length):
            seq_pred_list.append(seq_pred[index, :length, :])
        return torch.cat(seq_pred_list, dim=0)

    else:
        attn_list = []
        for index, length in enumerate(batch_length):
            seq_pred_list.append(seq_pred[index, :length, :])
            attn_list.append(attn[index, :length, :])
        return torch.cat(seq_pred_list, dim=0), torch.cat(attn_list, dim=0)


def get_fasta_statis_without_pathogen(max_seq_length=50, n_aa_type=20):
    fasta_path = "data/source/amp/amp_peptide_without_pathogen.fasta"
    seq_length = np.zeros(max_seq_length + 1)
    aa_count = np.zeros(n_aa_type)

    for record in SeqIO.parse(fasta_path, "fasta"):
        record_seq = record.seq
        length = len(record_seq)

        seq_length[length] += 1

        for aa in record_seq:
            aa_count[AA_dict[aa]] += 1

    seq_length = seq_length / sum(seq_length)
    aa_count = aa_count / sum(aa_count)
    return seq_length, aa_count


def get_fasta_statis_with_pathogen(n_aa_type=20):
    data = pd.read_csv(r'data/source/amp/peptide_pathogen_triple.csv')
    seq_list = list(data.sequence)
    pathogen_list = list(data.pathogen)

    data_triple = {}
    for index in range(len(seq_list)):
        if pathogen_list[index] in data_triple.keys():
            data_triple[pathogen_list[index]].append(seq_list[index])
        else:
            data_triple[pathogen_list[index]] = [seq_list[index]]

    result = {}
    for key in data_triple.keys():
        seq_list = data_triple[key]

        aa_count = np.zeros(n_aa_type)

        for seq in seq_list:
            for aa in seq:
                aa_count[AA_dict[aa]] += 1
        aa_count = aa_count / sum(aa_count)

        result[key] = aa_count

    return result


def save_output_seq(out_seq_list, file_name=None):
    record_list = []

    for i, seq_str in enumerate(out_seq_list):
        seq_id = "AMP_{}".format(i)
        seq_desc = ""

        record = SeqRecord(Seq(seq_str), id=seq_id, description=seq_desc)
        record_list.append(record)

    time_str = str(pd.Timestamp.now())[:16]

    if not os.path.exists("data/output/fasta"):
        os.makedirs("data/output/fasta")

    if file_name is not None:
        record_path = "data/output/fasta/{}.fasta".format(file_name)
    else:
        record_path = "data/output/fasta/AMP_{}.fasta".format(time_str)

    print("generate " + record_path)
    SeqIO.write(record_list, record_path, "fasta")

    return record_path


def get_fasta_statis(max_seq_length=50, n_aa_type=20, pathogen_type=None):
    if pathogen_type is not None:
        fasta_path = "data/source/amp/amp_peptide.fasta"
    else:
        fasta_path = "data/source/amp/amp_peptide_without_pathogen.fasta"

    seq_length = np.zeros(max_seq_length + 1)
    # aa_count = np.zeros(n_aa_type)

    for record in SeqIO.parse(fasta_path, "fasta"):
        record_seq = record.seq
        length = len(record_seq)
        seq_length[length] += 1

        # for aa in record_seq:
        #     aa_count[AA_dict[aa]] += 1

    seq_length = seq_length / sum(seq_length)
    # aa_count = aa_count / sum(aa_count)
    return seq_length
