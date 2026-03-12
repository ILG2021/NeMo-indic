import sherpa_onnx
import wave
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sherpa-ONNX Inference for IndicConformer")
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", type=str, required=True, help="Path to decoder.onnx")
    parser.add_argument("--joiner", type=str, required=True, help="Path to joiner.onnx")
    parser.add_argument("--tokens", type=str, required=True, help="Path to tokens.txt")
    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")
    args = parser.parse_args()

    # 初始化识别器
    # IndicConformer 是 Transducer (RNNT) 模型
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        tokens=args.tokens,
        num_threads=4,
        sample_rate=16000,
        feature_config=sherpa_onnx.FeatureConfig(sample_rate=16000, feature_dim=80),
        decoding_method="greedy_search", # 或者 modified_beam_search
    )

    # 读取音频
    with wave.open(args.wav, "rb") as f:
        assert f.getnchannels() == 1, "只支持单声道"
        assert f.getsampwidth() == 2, "采样位深需为 16bit"
        assert f.getframerate() == 16000, "采样率需为 16kHz"
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_np = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768

    # 推理
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples_np)
    recognizer.decode_stream(stream)
    
    print(f"识别结果: {stream.result.text}")

if __name__ == "__main__":
    main()
