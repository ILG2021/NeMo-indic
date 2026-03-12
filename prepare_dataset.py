import os
import json
import argparse
import librosa

def build_manifest(metadata_path, wav_dir, manifest_path):
    print(f"Reading metadata from {metadata_path}...")
    
    records = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 假设文件格式是：audio_filename.wav|标注文本
        parts = line.split('|')
        if len(parts) != 2:
            print(f"Skipping invalid line: {line}")
            continue
            
        filename, transcript = parts[0].strip(), parts[1].strip()
        wav_path = os.path.join(wav_dir, filename)
        
        if not os.path.exists(wav_path):
            print(f"Warning: Audio file not found: {wav_path}")
            continue
            
        # 获取音频时长
        duration = librosa.get_duration(path=wav_path)
        
        # 构造 NeMo manifest 对象
        record = {
            "audio_filepath": os.path.abspath(wav_path),
            "duration": duration,
            "text": transcript
        }
        records.append(record)
        
    print(f"Writing {len(records)} records to {manifest_path}...")
    with open(manifest_path, 'w', encoding='utf-8') as f_out:
        for record in records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LJSpeech metadata to NeMo manifest")
    parser.add_argument("--metadata", type=str, required=True, help="Path to LJSpeech metadata file (e.g., metadata.csv)")
    parser.add_argument("--wav-dir", type=str, required=True, help="Path to directory containing wav files")
    parser.add_argument("--manifest", type=str, required=True, help="Output manifest path (e.g., train_manifest.jsonl)")
    args = parser.parse_args()
    
    build_manifest(args.metadata, args.wav_dir, args.manifest)
