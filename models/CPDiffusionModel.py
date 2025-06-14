import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from models.KgTripleModel import get_pathogen_KG_feature_from_pretraining
from modules.sequence.encoder import SeqFFN, SeqNoise, ConditionFFN, AdapterFFN
from modules.sequence.transformer import SeqTransformer
from util.diffusion_util import get_para_schedule, get_Qt_weight, multinomial_kl, token_aa_acc, get_Qt_weight_batch
from util.embedding.sequence import get_emb_from_batch_logit, remove_padding, logit_to_index, index_to_fasta, \
    save_output_seq, get_fasta_statis


class KGCondDiffusion(pl.LightningModule):
    def __init__(
            self,
            n_class=20,
            # amino acid number
            n_seq_emb=185,
            # 185 = seq_ebm 121 + cond_emb 64
            n_cond_emb=128,
            n_hidden=512,
            # embedding for transformer input
            n_timestep=1000,
            n_self_atte_head=8,
            clamp=-50,
            beta_schedule="linear",
            beta_start=1.e-4,
            beta_end=2.e-2,
            max_len=50,
            # max length of peptide sequence
            learning_rate=1e-4,
            learning_rate_pre=None,
            condition_status=True,
            # whether using condition diffusion
            condition_weight=2
            # {0, 1, 1.5, 2, 3, 5, 7.5, 10}
    ):
        super().__init__()

        self.condition_status = condition_status
        self.condition_weight = condition_weight

        self.n_class = n_class
        self.n_timestep = n_timestep
        self.clamp = clamp
        self.n_self_atte_head = n_self_atte_head
        self.max_len = max_len

        betas, alphas, alphas_bar = get_para_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timestep=n_timestep,
            device=self.device
        )

        self.betas = nn.Parameter(betas, requires_grad=False)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.alphas_bar = nn.Parameter(alphas_bar, requires_grad=False)
        # diffusion parameters

        if learning_rate_pre is None:
            self.lr_pre = learning_rate
        else:
            self.lr_pre = learning_rate_pre

        self.lr = learning_rate

        self.seqTransformer = SeqTransformer(
            output_dim=n_hidden,
            n_emb=n_hidden,
            n_head=n_self_atte_head,
            diff_status=True
        )

        self.condition_ffn = ConditionFFN(n_cond_emb, n_cond_emb // 2)
        self.adapter_ffn = AdapterFFN(n_seq_emb, n_hidden)
        # pathogen adapter

        self.seq_ffn = SeqFFN(n_hidden, n_class)

        self.time_sampler = torch.distributions.Categorical(torch.ones(n_timestep))

        self.prior_noise = SeqNoise()

    def forward(self, x_t, time_steps, batch_index, batch_size, batch_length, condition):
        cond_emb = self.condition_ffn(condition)
        cond_emb = cond_emb.unsqueeze(1).repeat(1, self.max_len, 1)

        x_t_emb, padding_mask, _ = get_emb_from_batch_logit(
            seq_logit_list=x_t,
            batch_index=batch_index,
            batch_size=batch_size,
            max_len=self.max_len,
            device=self.device
        )

        x_t_emb = torch.cat([x_t_emb, cond_emb], dim=-1)
        x_t_emb = self.adapter_ffn(x_t_emb)

        seq_emb = self.seqTransformer(
            x_t_emb,
            time_step=time_steps,
            padding_mask=padding_mask
        )

        seq_emb = remove_padding(seq_emb, batch_length)
        output = self.seq_ffn(seq_emb)

        seq_pred = F.softmax(output, dim=-1).float()
        return seq_pred

    def get_loss(self, batch):
        n_seq = len(batch.seq)
        seq_time_steps = self.time_sampler.sample(sample_shape=torch.Size([n_seq])).to(self.device)

        alphas_bar = self.alphas_bar.index_select(0, seq_time_steps)
        x0_real = batch.x

        if self.condition_status:
            # condition diffusion
            condition_Qt_weight = get_Qt_weight_batch(
                alphas_bar,
                batch.condition_noise,
                batch.batch,
                self.device,
                self.n_class
            )

            condition_x_t = torch.matmul(x0_real.unsqueeze(1), condition_Qt_weight).squeeze(1)
            condition_kg_feature = batch.condition_feature

            condition_x0_pred = self.forward(
                condition_x_t,
                seq_time_steps,
                batch.batch,
                n_seq,
                batch.length,
                condition_kg_feature
            )

            condition_loss = multinomial_kl(condition_x0_pred, x0_real)
            self.log("condition_loss", condition_loss, prog_bar=True)

            if self.condition_weight != 0:
                normal_kg_feature = batch.normal_feature
                # condition of generic peptides

                normal_Qt_weight = get_Qt_weight_batch(
                    alphas_bar,
                    batch.normal_noise,
                    batch.batch,
                    self.device,
                    self.n_class
                )
                # normal_noise for amino acids

                normal_x_t = torch.matmul(x0_real.unsqueeze(1), normal_Qt_weight).squeeze(1)

                normal_x0_pred = self.forward(
                    normal_x_t,
                    seq_time_steps,
                    batch.batch,
                    n_seq,
                    batch.length,
                    normal_kg_feature
                )

                normal_loss = multinomial_kl(normal_x0_pred, x0_real)
                self.log("normal_loss", normal_loss, prog_bar=True)

                kl_loss = normal_loss + self.condition_weight * torch.abs(condition_loss - normal_loss)
            else:
                kl_loss = condition_loss

            pred_score = token_aa_acc(condition_x0_pred, x0_real, self.device)
            x0_pred = condition_x0_pred

        else:
            # non-condition diffusion
            # [batch_size, 20]
            uniform_noise = (self.prior_noise.get_noise(device=self.device,
                                                        uniform_state=True).repeat(alphas_bar.size(0), 1))
            # uniform_noise for amino acids

            normal_Qt_weight = get_Qt_weight_batch(
                alphas_bar,
                uniform_noise,
                batch.batch,
                self.device,
                self.n_class
            )

            normal_x_t = torch.matmul(x0_real.unsqueeze(1), normal_Qt_weight).squeeze(1)

            normal_x0_pred = self.forward(
                normal_x_t,
                seq_time_steps,
                batch.batch,
                n_seq,
                batch.length,
                condition=None
            )
            x0_pred = normal_x0_pred
            kl_loss = multinomial_kl(x0_pred, x0_real)
            pred_score = token_aa_acc(x0_pred, x0_real, self.device)

        self.log("kl_loss", kl_loss, prog_bar=True)
        self.log("pred_score", pred_score, prog_bar=True)

        return kl_loss, x0_pred, x0_real

    def training_step(self, batch, batch_idx):
        kl_loss, x0_pred, x0_real = self.get_loss(
            batch=batch
        )

        return {"loss": kl_loss, "x0_pred": x0_pred, "x0_real": x0_real}

    def training_epoch_end(self, training_step_outputs):

        epoch_kl_loss = torch.stack([step["loss"] for step in training_step_outputs])
        epoch_x0_pred = torch.concat([step["x0_pred"] for step in training_step_outputs], dim=0)
        epoch_x0_real = torch.concat([step["x0_real"] for step in training_step_outputs], dim=0)
        # calculate the average kl loss and sequence score for each epoch

        epoch_seq_pred_score = token_aa_acc(epoch_x0_pred, epoch_x0_real, self.device)

        self.log("total_seq_score", epoch_seq_pred_score, prog_bar=True)
        self.log("avg_kl_loss", epoch_kl_loss.mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.seqTransformer.parameters(), 'lr': self.lr_pre},
            # load pre-training learning rate

            {'params': self.adapter_ffn.parameters(), 'lr': self.lr},
            {'params': self.condition_ffn.parameters(), 'lr': self.lr},
            {'params': self.seq_ffn.parameters(), 'lr': self.lr},
        ])
        return optimizer

    def q_posterior(self, x0, time_step, noise, batch):
        """
        p_theta(xt_1|xt) = q(xt-1|xt,x0)*p(x0|xt)

        log(p_theta(xt_1|xt)) = log(q(xt-1|xt,x0)) + log(p(x0|xt))
                               = log(p(x0|xt)) + log(q(xt|xt-1,x0)) + log(q(xt-1|x0)) - log(q(xt|x0))

        Bayesian Rule: log(q(xt-1|xt,x0)) -> log(q(xt|xt-1,x0)) + log(q(xt-1|x0)) - log(q(xt|x0))
        """

        time_step = (time_step + (self.n_timestep + 1)) % (self.n_timestep + 1)

        alphas = self.alphas.index_select(0, time_step)
        alphas_bar_t = self.alphas_bar.index_select(0, time_step)
        alphas_bar_t_1 = self.alphas_bar.index_select(0, time_step - 1)

        # q(xt|x0)
        Qt_weight = get_Qt_weight(alphas_bar_t, noise, batch, self.device, self.n_class)
        xt_from_x0 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        # q(xt-1|x0)
        Qt_weight = get_Qt_weight(alphas_bar_t_1, noise, batch, self.device, self.n_class)
        xt_1_from_x0 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        # q(xt|xt_1,x0) -> q(xt|xt_1)q(xt_1|x_0)
        Qt_weight = get_Qt_weight(alphas, noise, batch, self.device, self.n_class)
        xt_from_xt_1 = torch.matmul(xt_1_from_x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        # log(p_theta(xt_1|xt)) = log(p(x0|xt)) - log(q(xt|x0)) + log(q(xt|xt-1,x0)) + log(q(xt-1|x0))
        xt_1_from_xt = torch.log(x0) - torch.log(xt_from_x0) + torch.log(xt_from_xt_1) + torch.log(xt_1_from_x0)
        xt_1_from_xt = torch.clamp(xt_1_from_xt, self.clamp, 0)
        xt_1_from_xt = torch.exp(xt_1_from_xt)
        # p_theta(xt_1|xt)

        return xt_1_from_xt

    @torch.no_grad()
    def denoise_seq_sample(self, n_seq=1, seq_length=None, fasta_out_statue: bool = False, pathogen_type=None,
                           file_name=None):
        seq_length_freq = get_fasta_statis(pathogen_type=pathogen_type)
        seq_freq = torch.tensor(seq_length_freq, device=self.device)
        D = torch.distributions.Categorical(seq_freq)
        # randomly choose the length of sequence from the frequency distribution (pathogen peptide length distribution)

        pathogen_kg_feature = get_pathogen_KG_feature_from_pretraining()
        # load pre-training pathogen KG feature

        out_seq_list = []
        out_seq_traj = []

        for i in range(n_seq):
            if seq_length is None:
                seq_len = D.sample()
            else:
                # seq_len = seq_length[i]
                seq_len = seq_length

            if pathogen_type is not None:
                noise = self.prior_noise.get_noise(device=self.device, pathogen_type=pathogen_type)
                condition_feature = pathogen_kg_feature[pathogen_type].to(self.device)
                # condition diffusion for pathogen
            else:
                noise = self.prior_noise.get_noise(device=self.device, uniform_state=True)
                condition_feature = None
                # non-condition diffusion

            seq_init = noise.repeat(seq_len, 1)
            # seq_index_t = logit_to_index(seq_init, random_state=True)
            seq_index_t = seq_init

            batch = torch.zeros(seq_len, device=self.device).long()
            t_list = torch.arange(self.n_timestep - 1, 0, -5).to(self.device)
            print("denoise {}-th sequence".format(i + 1))

            for time_steps in tqdm(t_list):
                # seq_emb = sequence_embedding(index=seq_index_t)
                token_time_steps = time_steps.repeat(seq_len)

                seq0_pred = self.forward(
                    seq_index_t,
                    time_steps,
                    batch,
                    1,
                    [seq_len],
                    condition_feature
                )

                seq_t_1 = self.q_posterior(
                    seq0_pred,
                    token_time_steps,
                    noise,
                    batch
                )

                seq_index_t = seq_t_1

                out_seq_traj.append(index_to_fasta(logit_to_index(seq_t_1, random_state=True)))

            seq_index_final = seq_index_t
            seq_index_final = logit_to_index(seq_index_final, random_state=False)
            seq_fasta = index_to_fasta(seq_index_final)

            out_seq_list.append(seq_fasta)

        if fasta_out_statue:
            record_path = save_output_seq(out_seq_list, file_name=file_name)
        else:
            record_path = None

        return out_seq_list, out_seq_traj, record_path
