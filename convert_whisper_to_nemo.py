import json
import argparse
import sys
import os

def convert_whisper_to_nemo(input_jsonl, output_jsonl):
    """
    Converts Whisper format jsonl to NeMo manifest format.
    Assumes the audio path in the input is already the absolute path.
    """
    print(f"Reading from {input_jsonl}...")
    
    records = []
    with open(input_jsonl, 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Extract required fields for NeMo
                audio_path = data.get("audio", {}).get("path", "")
                text = data.get("sentence", "")
                duration = data.get("duration", 0.0)
                
                # As user stated, paths are already absolute, so no joining needed.
                nemo_record = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": text
                }
                records.append(nemo_record)
                
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {idx + 1}: {line}")
            except Exception as e:
                print(f"Error processing line {idx + 1}: {e}")

    print(f"Writing {len(records)} records to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for record in records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper JSONL format to NeMo Manifest JSONL format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input Whisper jsonl file")
    
    args = parser.parse_args()
    
    input_path = args.input
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}-nemo{ext}"
    
    convert_whisper_to_nemo(input_path, output_path)
