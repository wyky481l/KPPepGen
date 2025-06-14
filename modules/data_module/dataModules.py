from Bio import SeqIO
from torch.utils.data import ConcatDataset, DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
from modules.data_module.dataSets import PeptideGoTripleDataSet, GoGoTripleDataSet, PepPathogenTripleDataSet, \
    PepSequenceDataSet, PepSequencePathogenDataSet, PepSequenceConditionDataSet


class GoGoTripleModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2048):
        super().__init__()
        self.batch_size = batch_size
        self.goGoTriple_dataset = GoGoTripleDataSet()

    def train_dataloader(self):
        return DataLoader(self.goGoTriple_dataset, batch_size=self.batch_size, shuffle=True)


class PeptideGoTripleModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1024):
        super().__init__()
        self.batch_size = batch_size

        self.peptideGoTriple_dataset = PeptideGoTripleDataSet()
        self.goGoTriple_dataset = GoGoTripleDataSet()
        self.concatDataset = ConcatDataset([self.peptideGoTriple_dataset, self.goGoTriple_dataset])

    def train_dataloader(self):
        return DataLoader(self.concatDataset, batch_size=self.batch_size, shuffle=True, num_workers=2)


class PeptidePathogenTripleModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1024):
        super().__init__()
        self.batch_size = batch_size

        self.peptidePathogenTriple_dataset = PepPathogenTripleDataSet()

    def train_dataloader(self):
        return DataLoader(self.peptidePathogenTriple_dataset, batch_size=self.batch_size, shuffle=True)


class PepSequenceDataModule(pl.LightningDataModule):
    def __init__(self, seq_pathogen_status: bool = True, batch_size: int = 1024):
        super().__init__()

        self.batch_size = batch_size
        _, self.fasta_list = get_fasta_list(seq_pathogen_status)
        self.dataset = PepSequenceDataSet(self.fasta_list)

    def train_dataloader(self):
        return pyg_DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class PepSequencePathogenDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1024):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = PepSequencePathogenDataSet()

    def train_dataloader(self):
        return pyg_DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class PepSequenceConditionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 512):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = PepSequenceConditionDataSet()

    def train_dataloader(self):
        return pyg_DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


def get_fasta_list(seq_pathogen_status):
    if seq_pathogen_status:
        fasta_path = "data/source/amp/amp_peptide.fasta"
    else:
        fasta_path = "data/source/amp/amp_peptide_without_pathogen.fasta"

    fasta_data = SeqIO.parse(fasta_path, "fasta")
    fasta_id_list = []
    fasta_seq_list = []

    for fasta in tqdm(fasta_data):
        fasta_id = fasta.id
        fasta_seq = str(fasta.seq)

        fasta_id_list.append(fasta_id)
        fasta_seq_list.append(fasta_seq)

    return fasta_id_list, fasta_seq_list
