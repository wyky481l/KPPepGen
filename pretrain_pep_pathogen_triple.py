# _*_ coding:utf-8 _*_
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from config import setting_pepPathogenTriple_init
from modules.data_module.dataModules import PeptidePathogenTripleModule
from models.KgTripleModel import PepPathogenSequenceModel
import pytorch_lightning as pl

if __name__ == "__main__":
    args = setting_pepPathogenTriple_init()
    pl.seed_everything(args.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        args.out_log_dir,
        name="pep_pathogen_Triple",
        version=None,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_checkpoint_dir,
        monitor='avg_loss',
        filename='pep_pathogen_Triple-{epoch:02d}-{avg_loss:.4f}',
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

    pepSeqGoModel_dict = torch.load(args.pepGoTriple_checkpoint)["state_dict"]
    # load the pre-training model for pep_go_triple;
    # like "data/output/pep_go_triple/pep_go_triple-XX.ckpt"
    pepSeqGoModel_paras = {k: v for k, v in pepSeqGoModel_dict.items() if 'seqTransformer' in k}

    model = PepPathogenSequenceModel(
        learning_rate_pre=args.learning_rate_pre,
        learning_rate=args.learning_rate,
        max_score=args.max_score,
        fasta_type=args.fasta_type
    )
    pepPathogenModel_dict = model.state_dict()
    pepPathogenModel_dict.update(pepSeqGoModel_paras)
    model.load_state_dict(pepPathogenModel_dict)

    dataModule = PeptidePathogenTripleModule(batch_size=args.batch_size)
    trainer.fit(model, dataModule)
