import json
import os
import shutil
import argparse
import sys

def fix_path_for_os(path):
    """
    Converts WSL paths (/mnt/e/...) to Windows paths (E:/...) if running on Windows.
    """
    if os.name == 'nt' and path.startswith('/mnt/'):
        # Convert /mnt/e/path to E:/path
        parts = path.split('/')
        drive = parts[2]
        remaining = "/".join(parts[3:])
        return f"{drive.upper()}:/{remaining}"
    return path

def fix_manifest_filenames(input_manifest):
    """
    Reads a NeMo manifest, detects long filenames, truncates them if necessary,
    and copies to a sibling 'wavs-rename' directory.
    """
    base, ext = os.path.splitext(input_manifest)
    output_manifest = f"{base}-fixed{ext}"
    
    processed_count = 0
    fix_count = 0
    created_dirs = set()

    print(f"Reading manifest: {input_manifest}")
    
    with open(input_manifest, 'r', encoding='utf-8') as f_in, \
         open(output_manifest, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                # This path might be WSL style (/mnt/e/...)
                raw_path = data.get("audio_filepath", "")
                if not raw_path:
                    continue

                # Adjust path if we are running on Windows
                old_path = fix_path_for_os(raw_path)
                
                # Get directory and filename
                dir_path = os.path.dirname(old_path)
                file_name = os.path.basename(old_path)
                
                # Setup rename directory
                if 'wavs' in dir_path:
                    rename_dir = dir_path.replace('wavs', 'wavs-rename')
                else:
                    rename_dir = os.path.join(dir_path, 'wavs-rename')
                
                if rename_dir not in created_dirs:
                    if not os.path.exists(rename_dir):
                        os.makedirs(rename_dir)
                    created_dirs.add(rename_dir)

                name_part, ext_part = os.path.splitext(file_name)
                
                # Check byte length (important: Linux limit is 255 BYTES TOTAL)
                # We check the full filename including extension
                filename_bytes = file_name.encode('utf-8')
                
                if len(filename_bytes) > 255:
                    # TRUNCATE LOGIC
                    name_part, ext_part = os.path.splitext(file_name)
                    ext_bytes = ext_part.encode('utf-8')
                    
                    # We need: len(name_part_bytes) + len(ext_bytes) <= 255
                    # Let's target 240 bytes for the whole thing to be safe
                    max_name_bytes = 240 - len(ext_bytes)
                    
                    name_part_bytes = name_part.encode('utf-8')
                    truncated_bytes = name_part_bytes[:max_name_bytes]
                    
                    while True:
                        try:
                            new_name_part = truncated_bytes.decode('utf-8')
                            break
                        except UnicodeDecodeError:
                            truncated_bytes = truncated_bytes[:-1]
                    new_file_name = new_name_part + ext_part
                    fix_count += 1
                    
                    # Target renamed path
                    if 'wavs' in dir_path:
                        rename_dir = dir_path.replace('wavs', 'wavs-rename')
                    else:
                        rename_dir = os.path.join(dir_path, 'wavs-rename')
                    
                    if rename_dir not in created_dirs:
                        if not os.path.exists(rename_dir):
                            os.makedirs(rename_dir)
                        created_dirs.add(rename_dir)
                        
                    new_full_path = os.path.join(rename_dir, new_file_name)
                    
                    # Physical Copy (Run this on Windows to bypass WSL limits!)
                    if os.path.exists(old_path):
                        if not os.path.exists(new_full_path):
                            shutil.copy2(old_path, new_full_path)
                    
                    # Convert back to WSL path for the manifest
                    # CRITICAL: Always use forward slashes for WSL manifest
                    manifest_path = new_full_path
                    if os.name == 'nt' and manifest_path.find(':/') != -1 or manifest_path.find(':\\') != -1:
                        # Normalize Windows path to use forward slashes
                        manifest_path = manifest_path.replace('\\', '/')
                        drive_letter = manifest_path[0].lower()
                        manifest_path = f"/mnt/{drive_letter}{manifest_path[2:]}"
                    else:
                        manifest_path = manifest_path.replace('\\', '/')
                else:
                    # KEEPS ORIGINAL
                    manifest_path = raw_path

                data["audio_filepath"] = manifest_path
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"Error on line {idx+1}: {e}")

    print(f"\nFinished!")
    print(f"Processed: {processed_count} lines")
    print(f"Truncated: {fix_count} filenames")
    print(f"New manifest: {output_manifest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix long filenames in NeMo manifests by truncating bytes and copying files.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input .jsonl manifest")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        fix_manifest_filenames(args.input)
    else:
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
