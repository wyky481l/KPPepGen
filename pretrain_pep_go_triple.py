# _*_ coding:utf-8 _*_
import torch
from config import setting_pepGoTriple_init
from models.KgTripleModel import PepGoSequenceModel
from modules.data_module.dataModules import PeptideGoTripleModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

if __name__ == "__main__":
    args = setting_pepGoTriple_init()
    pl.seed_everything(args.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        args.out_log_dir,
        name="pep_go_triple",
        version=None,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_checkpoint_dir,
        monitor='avg_loss',
        filename='pep_go_triple-{epoch:02d}-{avg_loss:.4f}-{pep_avg_loss:.4f}-{go_avg_loss:.4f}',
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

    goGoModel_dict = torch.load(args.goGoTriple_checkpoint)["state_dict"]
    # load the pre-training model for go_go_triple; like "data/output/go_go_triple/go_go_triple-XX.ckpt"
    goTermProjector_paras = {k: v for k, v in goGoModel_dict.items() if 'goTermProjector' in k}
    relationProjector_paras = {k: v for k, v in goGoModel_dict.items() if 'relationProjector' in k}

    model = PepGoSequenceModel(
        learning_rate_seq=args.learning_rate_seq,
        learning_rate_go=args.learning_rate_go,
        max_score=args.max_score,
        fasta_type=args.fasta_type
    )

    pepGoModel_dict = model.state_dict()
    pepGoModel_dict.update(goTermProjector_paras)
    pepGoModel_dict.update(relationProjector_paras)
    model.load_state_dict(pepGoModel_dict)

    dataModule = PeptideGoTripleModule(batch_size=args.batch_size)
    trainer.fit(model, dataModule)
