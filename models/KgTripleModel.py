import os
import pickle
import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
from modules.sequence.encoder import get_sentence_emb
from modules.sequence.transformer import SeqTransformer
from modules.triple.encode import transe_loss, SeqTermGOProjector, GoTermGOProjector, RelationGOProjector, \
    PathogenTermProjector, SeqTermPathogenProjector
from util.embedding.go_terms import get_go_prefeature, get_go_feature_from_term_list
from util.embedding.sequence import get_peptide_fasta, get_feature_from_batch_sequence, head_list_sort, \
    get_seq_feature_from_dataset, get_fasta_statis_with_pathogen
from util.embedding.pathogen import get_pathogen_prefeature, get_pathogen_triple, get_pathogen_feature_from_term_list


class GOGoModel(pl.LightningModule):
    def __init__(self, max_score=12, learning_rate=1e-4):
        super().__init__()

        self.go_prefeature_dict = get_go_prefeature()
        self.goTermProjector = GoTermGOProjector()
        self.relationProjector = RelationGOProjector()

        self.max_score = max_score
        self.lr = learning_rate

    def forward(self, batch):
        head_go_id, relation_feature, tail_go_id, neg_go_id, head_type = batch

        head_go_id = np.array(head_go_id)

        head_go_feature = get_go_feature_from_term_list(head_go_id, self.go_prefeature_dict, self.device)
        # [N,768] embedding_size=768 from PubMedBert
        tail_go_feature = get_go_feature_from_term_list(tail_go_id, self.go_prefeature_dict, self.device)
        neg_go_feature = get_go_feature_from_term_list(neg_go_id, self.go_prefeature_dict, self.device)

        head_go_feature = self.goTermProjector(head_go_feature)
        tail_go_feature = self.goTermProjector(tail_go_feature)
        neg_go_feature = self.goTermProjector(neg_go_feature)

        relation_feature = torch.squeeze(relation_feature, dim=1)
        relation_feature = self.relationProjector(relation_feature)

        return head_go_feature, relation_feature, tail_go_feature, neg_go_feature

    def get_loss(self, batch):
        head_go_feature, relation_feature, tail_go_feature, neg_go_feature = self.forward(batch)

        total_loss, _, _ = transe_loss(
            head_go_feature,
            relation_feature,
            tail_go_feature,
            neg_go_feature,
            self.max_score
        )

        self.log("train_loss", total_loss, prog_bar=True)

        return {"loss": total_loss}

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(
            batch=batch
        )
        return loss

    def training_epoch_end(self, training_step_outputs):
        epoch_loss_list = torch.stack([step["loss"] for step in training_step_outputs])
        epoch_avg_loss = epoch_loss_list.mean()

        self.log("avg_loss", epoch_avg_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer


class PepGoSequenceModel(pl.LightningModule):
    def __init__(
            self,
            input_seq_dim=121,
            input_go_terms_dim=768,
            input_relation_dim=6,
            output_dim=128,
            n_hidden_dim=512,
            n_self_atte_head=8,
            max_score=10,
            learning_rate_seq=1e-3,
            learning_rate_go=1e-3,
            fasta_type="uniport",
            max_len=50
    ):
        super().__init__()

        self.max_score = max_score
        self.max_len = max_len
        self.lr_seq = learning_rate_seq
        self.lr_go = learning_rate_go
        self.n_self_atte_head = n_self_atte_head

        # load pre-processed data
        self.fasta_dict = get_peptide_fasta(fasta_type)
        # all the peptide sequences in the dataset

        self.seq_feature_dict = get_seq_feature_from_dataset(fasta_type, bio_status=True)
        # all the sequence features in the dataset

        self.go_prefeature_dict = get_go_prefeature()
        # loading the pre-processed go features from PubMedBert

        self.seqTransformer = SeqTransformer(
            output_dim=n_hidden_dim,
        )

        self.seqTermProjector = SeqTermGOProjector(
            input_dim=n_hidden_dim,
            output_dim=output_dim
        )
        self.goTermProjector = GoTermGOProjector(
            input_dim=input_go_terms_dim,
            output_dim=output_dim
        )
        self.relationProjector = RelationGOProjector(
            input_dim=input_relation_dim,
            output_dim=output_dim
        )
        self.input_layer = nn.Linear(
            input_seq_dim,
            n_hidden_dim
        )

    def forward(self, batch):
        head_go_id, relation_feature, tail_go_id, neg_go_id, head_type_list = batch
        head_type_list = head_type_list.cpu().numpy()
        head_go_id = np.array(head_go_id)

        head_pep_list = head_go_id[head_type_list == 1]
        sequence_input, padding_mask, seq_len_list = get_feature_from_batch_sequence(
            head_pep_list,
            self.seq_feature_dict,
            max_len=self.max_len,
            device=self.device
        )

        x_emb = self.input_layer(sequence_input)
        seq_output_feature = self.seqTransformer(
            x_emb=x_emb,
            time_step=None,
            padding_mask=padding_mask
        )

        sentence_emb = get_sentence_emb(seq_output_feature, seq_len_list)
        head_seq_feature = self.seqTermProjector(sentence_emb)

        relation_feature = torch.squeeze(relation_feature, dim=1)
        relation_feature = self.relationProjector(relation_feature)

        head_go_list = head_go_id[head_type_list == 0]

        if len(head_go_list) > 0:
            head_go_feature = get_go_feature_from_term_list(head_go_list, self.go_prefeature_dict, self.device)
            head_go_feature = self.goTermProjector(head_go_feature)

            head_feature = head_list_sort(head_seq_feature, head_go_feature, head_type_list)
        else:
            head_feature = head_seq_feature

        tail_go_feature = get_go_feature_from_term_list(tail_go_id, self.go_prefeature_dict, self.device)
        neg_go_feature = get_go_feature_from_term_list(neg_go_id, self.go_prefeature_dict, self.device)

        tail_go_feature = self.goTermProjector(tail_go_feature)
        neg_go_feature = self.goTermProjector(neg_go_feature)

        return head_feature, relation_feature, tail_go_feature, neg_go_feature, head_type_list

    def get_loss(self, batch):
        head_feature, relation_feature, tail_go_feature, neg_go_feature, head_type_list = self.forward(batch)

        total_loss, pos_loss_list, neg_loss_list = transe_loss(
            head_feature,
            relation_feature,
            tail_go_feature,
            neg_go_feature,
            self.max_score
        )

        pep_loss = pos_loss_list[head_type_list == 1].mean() + neg_loss_list[head_type_list == 1].mean()
        go_loss = pos_loss_list[head_type_list == 0].mean() + neg_loss_list[head_type_list == 0].mean()

        self.log("train_loss", total_loss, prog_bar=True)

        return {"loss": total_loss, "pep_loss": pep_loss, "go_loss": go_loss}

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(
            batch=batch
        )
        return loss

    def training_epoch_end(self, training_step_outputs):
        epoch_loss_list = torch.stack([step["loss"] for step in training_step_outputs])
        epoch_avg_loss = epoch_loss_list.mean()

        epoch_pep_loss_list = torch.stack([step["pep_loss"] for step in training_step_outputs])
        epoch_go_loss_list = torch.stack([step["go_loss"] for step in training_step_outputs])

        epoch_pep_avg_loss = epoch_pep_loss_list.mean()
        epoch_go_avg_loss = epoch_go_loss_list.mean()

        self.log("avg_loss", epoch_avg_loss, prog_bar=True)
        self.log("pep_avg_loss", epoch_pep_avg_loss, prog_bar=True)
        self.log("go_avg_loss", epoch_go_avg_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            # seq term training
            {'params': self.seqTransformer.parameters(), 'lr': self.lr_seq},
            {'params': self.seqTermProjector.parameters(), 'lr': self.lr_seq},

            # go term with pre-training
            {'params': self.goTermProjector.parameters(), 'lr': self.lr_go},
            {'params': self.relationProjector.parameters(), 'lr': self.lr_go},

        ])
        return optimizer


class PepPathogenSequenceModel(pl.LightningModule):
    def __init__(
            self,
            input_seq_dim=121,
            input_pathogen_terms_dim=768,
            output_dim=128,
            n_hidden_dim=512,
            n_self_atte_head=8,
            max_score=10,
            learning_rate_pre=None,
            learning_rate=1e-4,
            fasta_type="amp",
            max_len=50,
    ):
        super().__init__()

        self.max_len = max_len
        self.max_score = max_score

        if learning_rate_pre is None:
            self.lr_pre = learning_rate
        else:
            self.lr_pre = learning_rate_pre

        self.lr = learning_rate
        self.n_self_atte_head = n_self_atte_head

        self.fasta_dict = get_peptide_fasta(fasta_type)
        self.seq_feature_dict = get_seq_feature_from_dataset(fasta_type)

        self.pathogen_prefeature_dict = get_pathogen_prefeature()
        self.pathogen_triple_list = get_pathogen_triple()

        self.seqTransformer = SeqTransformer(
            output_dim=n_hidden_dim
        )

        self.seqTermProjector = SeqTermPathogenProjector(
            input_dim=n_hidden_dim,
            output_dim=output_dim
        )

        self.pathogenTermProjector = PathogenTermProjector(
            input_dim=input_pathogen_terms_dim,
            output_dim=output_dim
        )

        self.input_layer = nn.Linear(input_seq_dim, n_hidden_dim)

    def forward(self, batch):
        head_pep_list, tail_pathogen_list, neg_pathogen_list = batch

        head_pathogen_list, tail_type_list, neg_type_list = self.pathogen_triple_list

        sequence_input, padding_mask, seq_len_list = get_feature_from_batch_sequence(
            head_pep_list,
            self.seq_feature_dict,
            max_len=self.max_len,
            device=self.device
        )

        x_emb = self.input_layer(sequence_input)
        seq_output_feature = self.seqTransformer(x_emb=x_emb, time_step=None, padding_mask=padding_mask)

        sentence_emb = get_sentence_emb(seq_output_feature, seq_len_list)
        head_seq_feature = self.seqTermProjector(sentence_emb)

        tail_pathogen_feature = get_pathogen_feature_from_term_list(tail_pathogen_list, self.pathogen_prefeature_dict,
                                                                    self.device)
        neg_pathogen_feature = get_pathogen_feature_from_term_list(neg_pathogen_list, self.pathogen_prefeature_dict,
                                                                   self.device)

        head_pathogen_feature = get_pathogen_feature_from_term_list(head_pathogen_list, self.pathogen_prefeature_dict,
                                                                    self.device)
        tail_type_feature = get_pathogen_feature_from_term_list(tail_type_list, self.pathogen_prefeature_dict,
                                                                self.device)
        neg_type_feature = get_pathogen_feature_from_term_list(neg_type_list, self.pathogen_prefeature_dict,
                                                               self.device)

        tail_pathogen_feature = self.pathogenTermProjector(tail_pathogen_feature)
        neg_pathogen_feature = self.pathogenTermProjector(neg_pathogen_feature)

        head_pathogen_feature = self.pathogenTermProjector(head_pathogen_feature)
        tail_type_feature = self.pathogenTermProjector(tail_type_feature)
        neg_type_feature = self.pathogenTermProjector(neg_type_feature)

        head_feature = torch.cat((head_seq_feature, head_pathogen_feature), dim=0)
        tail_feature = torch.cat((tail_pathogen_feature, tail_type_feature), dim=0)
        neg_tail_feature = torch.cat((neg_pathogen_feature, neg_type_feature), dim=0)

        return head_feature, tail_feature, neg_tail_feature

    def get_loss(self, batch):
        head_feature, tail_feature, neg_feature = self.forward(batch)

        total_loss, pos_loss_list, neg_loss_list = transe_loss(
            head=head_feature,
            relation=None,
            tail=tail_feature,
            neg_tail=neg_feature,
            max_score=self.max_score
        )
        return {"loss": total_loss}

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(
            batch=batch
        )
        return loss

    def training_epoch_end(self, training_step_outputs):
        epoch_loss_list = torch.stack([step["loss"] for step in training_step_outputs])
        epoch_avg_loss = epoch_loss_list.mean()

        self.log("avg_loss", epoch_avg_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.seqTransformer.parameters(), 'lr': self.lr_pre},

            {'params': self.seqTermProjector.parameters(), 'lr': self.lr},
            {'params': self.pathogenTermProjector.parameters(), 'lr': self.lr},

        ])
        return optimizer

    def get_pathogen_output(self, pathogen_feature):
        pathogen_feature_output = self.pathogenTermProjector(pathogen_feature)
        return pathogen_feature_output


def get_pathogen_KG_feature_from_pretraining(model_path=None):
    path = "data/source/amp/pathogen_KG_prefeature.pkl"
    # model_path = "data/output/pep_pathogen_triple/pep_pathogen_Triple-epoch=XX-avg_loss=XX.ckpt"

    if os.path.exists(path):
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result

    assert model_path is not None, "Please input the path of pre-training model for pep_pathogen_triple"

    pathogen_list = list(get_fasta_statis_with_pathogen().keys())
    pathogen_feature_dict = get_pathogen_prefeature()

    model = PepPathogenSequenceModel()
    model.load_state_dict(
        state_dict=torch.load(
            model_path,
            map_location=torch.device('cpu')
        )["state_dict"],
        strict=False)

    pathogen_KG_prefeature = {}

    for pathogen in pathogen_list:
        feature = pathogen_feature_dict[pathogen][2]
        KG_feature = model.get_pathogen_output(torch.tensor(feature))
        pathogen_KG_prefeature[pathogen] = KG_feature

    with open(path, "wb") as f:
        pickle.dump(pathogen_KG_prefeature, f)

    return pathogen_KG_prefeature
