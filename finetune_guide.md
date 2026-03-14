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
python finetune.py \
    --model_path "path/to/indicconformer_stt_hi_hybrid_rnnt_large.nemo" \
    --train_manifest train_manifest.jsonl \
    --val_manifest val_manifest.jsonl \
    --epochs 10 \
    --batch_size 4 \
    --lr 5e-6
```
> [!TIP]
> IndicConformer 模型体积较大（如 Large 或 600M），如果显存不足，请降低 `batch_size`。

## 第三步：导出为 Sherpa-ONNX 格式

由于您使用的是 **IndicConformer (Hybrid RNNT)** 且需要用 `sherpa-onnx` 推理，需要将 `.nemo` 导出为 `encoder.onnx`, `decoder.onnx`, `joiner.onnx` 以及 `tokens.txt`。

### 1. 安装导出工具
推荐使用 `sherpa-onnx` 附带的脚本进行精确转换：
```bash
pip install sherpa-onnx
git clone https://github.com/k2-fsa/sherpa-onnx
```

### 2. 执行导出
```bash
python sherpa-onnx/python-api-examples/export-nemo-transducer-to-onnx.py \
    --input finetuned_model.nemo \
    --output_dir ./sherpa_model
```
该命令会自动处理模型权重剥离、Tokenizer 提取和 ONNX 导出。

## 第四步：使用 Sherpa-ONNX 推理

导出后，你会得到一个包含多个 `.onnx` 和 `tokens.txt` 的文件夹。使用 `sherpa_inference.py` 进行高性能推理：

```bash
python sherpa_inference.py \
    --encoder ./sherpa_model/encoder.onnx \
    --decoder ./sherpa_model/decoder.onnx \
    --joiner ./sherpa_model/joiner.onnx \
    --tokens ./sherpa_model/tokens.txt \
    --wav test_audio.wav
```

### 为什么选择 Sherpa-ONNX？
1. **无 Python 依赖**：可在 C++, Android, iOS 等环境运行。
2. **内存占用低**：适合嵌入式和边缘侧部署。
3. **支持流式识别**：IndicConformer 虽然是大模型，但架构支持高效的量化和推理。
