import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.utils import logging
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    # 1. 恢复预训练模型
    logging.info(f"Loading model from: {args.model_path}")
    
    # 检查是否是本地文件
    if args.model_path.endswith('.nemo'):
        # IndicConformer 通常是 ASRModel 子类 (例如 EncDecRNNTHybridModel)
        try:
            model = nemo_asr.models.ASRModel.restore_from(args.model_path)
        except Exception as e:
            logging.error(f"Failed to load model from {args.model_path}: {e}")
            return
    else:
        # 如果提供了模型名称则尝试从云端加载
        try:
            model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.model_path)
        except:
            model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=args.model_path)

    # 2. 修改配置以匹配我们的数据集
    model.cfg.train_ds.manifest_filepath = [args.train_manifest]
    model.cfg.train_ds.batch_size = args.batch_size
    model.cfg.validation_ds.manifest_filepath = [args.val_manifest]
    model.cfg.validation_ds.batch_size = args.batch_size

    # 设置较小的微调学习率
    model.cfg.optim.lr = args.lr

    # 更新模型以使用新的数据配置
    model.setup_training_data(train_data_config=model.cfg.train_ds)
    model.setup_validation_data(val_data_config=model.cfg.validation_ds)

    # 3. 配置 PyTorch Lightning Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        save_top_k=1,
        verbose=True,
        monitor="val_wer",
        mode="min",
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=10
    )

    # 将 Trainer 传入模型
    model.set_trainer(trainer)

    # 4. 开始训练
    logging.info("Starting training...")
    trainer.fit(model)
    
    # 5. 保存最终模型
    model.save_to("finetuned_model.nemo")
    logging.info("Finetuned model saved to finetuned_model.nemo")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Finetune NeMo ASR model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .nemo file or pretrained model name")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    args = parser.parse_args()
    main(args)
