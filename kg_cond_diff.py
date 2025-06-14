# _*_ coding:utf-8 _*_
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from config import setting_pepGen_init
from models.CPDiffusionModel import KGCondDiffusion
from models.KgTripleModel import get_pathogen_KG_feature_from_pretraining
from modules.data_module.dataModules import PepSequenceConditionDataModule
import pytorch_lightning as pl

# prompt diffusion with the pre-training knowledge graph
if __name__ == "__main__":
    args = setting_pepGen_init()
    pl.seed_everything(args.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        args.out_log_dir,
        name="pepSeqCondDiff",
        version=None,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_checkpoint_dir,
        monitor='total_seq_score',
        filename='pepSeqCondDiff-{epoch:02d}-{avg_kl_loss:.5f}-{total_seq_score:.4f}',
        save_top_k=args.save_top_k,
        mode=args.save_mode,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.n_max_epochs,
        gpus=args.gpus,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    # load pre-trained knowledge model with sequence transformer
    pepSeqPathogenModel_dict = torch.load(
        args.pepPathogenTriple_checkpoint,
        map_location=torch.device('cuda:0'))[
        "state_dict"]
    pepSeqPathogenModel_paras = {k: v for k, v in pepSeqPathogenModel_dict.items() if 'seqTransformer' in k}
    # extract the key transformer parameters from pre-trained model

    get_pathogen_KG_feature_from_pretraining(model_path=args.pepPathogenTriple_checkpoint)
    # extract the pathogen prompts from pre-trained knowledge graph for diffusion process;

    model = KGCondDiffusion(
        n_timestep=args.n_timestep,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        learning_rate_pre=args.learning_rate_pre,
        learning_rate=args.learning_rate,
        condition_weight=args.condition_weight
    )
    KGCondDiffusion_dict = model.state_dict()
    KGCondDiffusion_dict.update(pepSeqPathogenModel_paras)
    model.load_state_dict(KGCondDiffusion_dict)

    dataModule = PepSequenceConditionDataModule(batch_size=args.batch_size)
    trainer.fit(model, dataModule)
