# _*_ coding:utf-8 _*_
import torch
from config import setting_goGoTriple_init
from models.KgTripleModel import PepGoSequenceModel, GOGoModel
from modules.data_module.dataModules import PeptideGoTripleModule, GoGoTripleModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

if __name__ == "__main__":
    args = setting_goGoTriple_init()
    pl.seed_everything(args.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        args.out_log_dir,
        name="go_go_triple",
        version=None,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_checkpoint_dir,
        monitor='avg_loss',
        filename='go_go_triple-{epoch:02d}-{avg_loss:.4f}',
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

    model = GOGoModel(
        max_score=args.max_score,
        learning_rate=args.learning_rate
    )

    dataModule = GoGoTripleModule(batch_size=args.batch_size)
    trainer.fit(model, dataModule)
