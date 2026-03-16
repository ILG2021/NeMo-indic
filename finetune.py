import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

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
    with open_dict(model.cfg):
        # 基础数据配置
        model.cfg.train_ds.manifest_filepath = [args.train_manifest]
        model.cfg.train_ds.batch_size = args.batch_size
        model.cfg.train_ds.num_workers = 4
        
        model.cfg.validation_ds.manifest_filepath = [args.val_manifest]
        model.cfg.validation_ds.batch_size = args.batch_size
        model.cfg.validation_ds.num_workers = 0

        # 【核心修复】强制指定所有数据层的通道和采样率
        for ds_cfg in [model.cfg.train_ds, model.cfg.validation_ds]:
            ds_cfg.sample_rate = 16000
            if 'force_channel' not in ds_cfg:
                ds_cfg.force_channel = "mono" # 如果没有，强行注入
            else:
                ds_cfg.force_channel = "mono" # 如果有，覆盖为 mono
            
            # 部分复杂模型可能在内部嵌套了这些参数，我们递归处理
            if 'parser_cfg' in ds_cfg:
                with open_dict(ds_cfg.parser_cfg):
                    ds_cfg.parser_cfg.force_channel = "mono"

        # 设置较小的微调学习率
        model.cfg.optim.lr = args.lr

    # 重新触发数据设置
    model.setup_training_data(train_data_config=model.cfg.train_ds)
    model.setup_validation_data(val_data_config=model.cfg.validation_ds)

    # 3. 配置 PyTorch Lightning Trainer
    # 设置每过多少 steps 进行一次验证并保存权重 (由 args.val_check_interval 控制)
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="nemo-asr-{epoch:02d}-{step}-{val_wer:.4f}",
        save_top_k=args.save_top_k,
        verbose=True,
        monitor="val_wer",
        mode="min",
        save_last=True, # 同时保存最新的一个，防止中断
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=args.epochs,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=1.0, # 梯度裁剪，防止 Loss 飙升
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=10
    )

    # 将 Trainer 传入模型
    model.set_trainer(trainer)

    # 4. 开始训练
    logging.info("Starting training...")
    if args.resume_path:
        logging.info(f"Resuming from checkpoint: {args.resume_path}")
        trainer.fit(model, ckpt_path=args.resume_path)
    else:
        trainer.fit(model)
    
    # 5. 保存最终模型
    model.save_to("finetuned_model.nemo")
    logging.info("Finetuned model saved to finetuned_model.nemo")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Finetune NeMo ASR model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .nemo file or pretrained model name")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to .ckpt file to resume training from")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val_check_interval", type=float, default=5000.0, help="If float < 1.0: fraction of epoch. If >= 1.0: number of steps.")
    parser.add_argument("--save_top_k", type=int, default=10, help="Save top K best models")
    
    args = parser.parse_args()
    main(args)
