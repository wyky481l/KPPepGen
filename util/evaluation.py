import os
import numpy as np
from Bio.Align import substitution_matrices
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
from tqdm import tqdm
from Bio import SeqIO, Align


def match_score(fasta, ref_path):
    # ref_path = "data/source/fasta/AMP.fasta"
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    score_list = []

    for record in SeqIO.parse(ref_path, "fasta"):
        amp_str = record.seq
        alignments = aligner.align(amp_str, fasta)
        score = alignments.score

        score_list.append(score)

    score_list = np.stack(score_list)
    return score_list.mean()


def match_score_batch(fasta_path, ref_path):
    fasta_list = list(SeqIO.parse(fasta_path, "fasta"))
    record_list = []

    for index in tqdm(range(len(fasta_list))):
        fasta_id = fasta_list[index].id
        fasta_str = fasta_list[index].seq
        score = match_score(fasta_str, ref_path)

        record = {"id": fasta_id, "score": score}
        record_list.append(record)

    return record_list


def fasta_aa_count(fasta_path):
    # fasta_path = "data/source/amp/amp_peptide.fasta"
    fasta_list = list(SeqIO.parse(fasta_path, "fasta"))
    result = [[] for _ in range(50)]
    for i in tqdm(range(len(fasta_list))):
        fasta = str(fasta_list[i].seq)

        for index in range(len(fasta)):
            aa = fasta[index]
            result[index].append(aa)

    aa_dict = [{} for _ in range(50)]
    for index in range(len(result)):
        for aa in result[index]:
            if aa not in aa_dict[index].keys():
                aa_dict[index][aa] = 0
            else:
                aa_dict[index][aa] = aa_dict[index][aa] + 1

    for aa in aa_dict:
        total_num = sum(aa.values())
        for key in aa.keys():
            aa[key] = aa[key] / total_num

    return aa_dict


# https://doi.org/10.1016/j.jmb.2006.09.020
# https://doi.org/10.1093/bioinformatics/btx285 modlamp
def ez_score(fasta, window=10):
    AMP = PeptideDescriptor(fasta, 'Ez')
    AMP.calculate_global(window)
    score = AMP.descriptor
    return score.squeeze()


# https://doi.org/10.1110/ps.062286306
def TM_tend_score(fasta, window=7):
    AMP = PeptideDescriptor(fasta, 'TM_tend')
    AMP.calculate_global(window)
    score = AMP.descriptor

    return score.squeeze()


def zdock_score(dir_path):
    files = os.listdir(dir_path)
    score_list = []
    for file in files:
        file_path = f'{dir_path}/{file}'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            deal_str = data[5]
            score = float(deal_str.split("\t")[-1])
            score_list.append(score)

    score_list = np.array(score_list)
    print(score_list.mean())


def instability_score(fasta):
    desc = GlobalDescriptor(fasta)
    desc.instability_index()
    score = desc.descriptor
    return score.squeeze()


def isoelectric_point_score(fasta):
    desc = GlobalDescriptor(fasta)
    desc.isoelectric_point()
    score = desc.descriptor
    return score.squeeze()


def calculate_charge(fasta):
    AMP = GlobalDescriptor(fasta)
    AMP.calculate_charge(ph=7, amide=False, append=False)
    score = AMP.descriptor
    return score.squeeze()


def hydrophobic_ratio(fasta):
    AMP = GlobalDescriptor(fasta)
    AMP.hydrophobic_ratio()
    score = AMP.descriptor
    return score.squeeze()


def aromaticity_score(fasta):
    desc = GlobalDescriptor(fasta)
    desc.aromaticity()
    score = desc.descriptor
    return score.squeeze()

