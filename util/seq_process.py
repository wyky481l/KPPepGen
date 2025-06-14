import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from util.constant import AA_type

def random_sequence(seq_num):
    length_freq = torch.ones(50)
    length_freq[:5] = 0
    aa_freq = torch.ones(20)

    length_sampler = torch.distributions.Categorical(length_freq)
    aa_sampler = torch.distributions.Categorical(aa_freq)
    # data = length_sampler.sample()

    sequence_list = []

    for _ in tqdm(range(seq_num)):
        seq_len = length_sampler.sample()
        seq = []
        for _ in torch.arange(0, seq_len):
            aa_index = aa_sampler.sample()
            seq.append(AA_type[aa_index])

        sequence_list.append("".join(seq))

    record_list = []
    seq_id = 1
    for seq in sequence_list:
        record = SeqRecord(Seq(seq),
                           id=f"seq_{seq_id}",
                           description="")
        seq_id = seq_id + 1
        record_list.append(record)

    SeqIO.write(record_list, "data/source/random.fasta", "fasta")