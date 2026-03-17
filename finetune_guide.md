# NeMo ASR 微调指南

本指南将指导你如何使用 Nvidia NeMo 框架基于你的 LJSpeech 格式数据（包含后缀的文件名|标注文本）微调 ASR 模型。整个过程包含两步：**数据格式转换** 和 **模型微调**。

## 第一步：准备 NeMo Manifest 数据格式

NeMo 要求 ASR 数据为 `JSONL` (JSON Lines) 格式，每一行是一个包含音频路径、时长和文本的 JSON 对象。我们首先使用提供的 `prepare_dataset.py` 将你的 LJSpeech 数据转换为这种格式。

### 1. 准备你的数据
假设你的数据文件格式如下（以 `|` 分隔）：
```text
audio_1.wav|你好，今天天气真不错
audio_2.wav|我们开始测试语音识别
```

### 2. 运行数据转换脚本
```bash
python prepare_dataset.py \
    --metadata path/to/metadata.csv \
    --wav-dir path/to/wav_folder \
    --manifest train_manifest.jsonl
```

## 第二步：运行微调

当你有了 `train_manifest.jsonl`，就可以运行微调脚本 `finetune.py`。
由于你是使用 AI4Bharat 的 **IndicConformer** 模型，请确保已经下载了对应的 `.nemo` 权重文件。

### 0. 环境特别说明
IndicConformer 建议使用 AI4Bharat 分支的 NeMo：
```bash
bash reinstall.sh
```

### 1. 运行微调脚本
你需要提供下载好的 `.nemo` 文件路径：
下载地址：
https://github.com/AI4Bharat/IndicConformerASR

```bash
python examples/asr/speech_to_text_finetune.py \
    init_from_nemo_model="indicconformer_stt_ne_hybrid_rnnt_large.nemo" \
    model.train_ds.manifest_filepath=\[\"train-fixed.jsonl\"\] \
    model.validation_ds.manifest_filepath=\[\"val-fixed.jsonl\"\] \
    model.train_ds.batch_size=4 \
    model.validation_ds.batch_size=4 \
    model.train_ds.num_workers=4 \
    model.validation_ds.num_workers=0 \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=3 \
    trainer.val_check_interval=5000 \
    exp_manager.exp_dir="finetune_experiments" \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.save_top_k=5 \
    exp_manager.checkpoint_callback_params.monitor="val_wer" \
    exp_manager.checkpoint_callback_params.mode="min" \
    exp_manager.resume_if_exists=True
```
> [!TIP]
> IndicConformer 模型体积较大（如 Large 或 600M），如果显存不足，请降低 `batch_size`。

## 第三步：导出为 Sherpa-ONNX 格式


