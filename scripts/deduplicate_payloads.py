import os
import json
import re

# --- CONFIGURATION ---
# Set this to False to ACTUALLY delete files. 
# While True, it will only print what would happen.
DRY_RUN = True 

# Path to your payloads folder (relative to this script)
# Assuming this script is in /extensions/sd-webui-api-payload-display/scripts/
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PAYLOADS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "payloads")

def get_file_timestamp(filename):
    """
    Extracts timestamp from filename for sorting.
    Format expected: payload_..._YYYYMMDD_HHMMSS.json
    Returns 0 if not found.
    """
    match = re.search(r"(\d{8}_\d{6})", filename)
    if match:
        return match.group(1)
    return "0"

def get_prompts_from_file(filepath):
    """
    Reads a JSON file and returns a tuple: (positive_prompt, negative_prompt)
    Returns None if the file is invalid or unreadable.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract prompts, defaulting to empty string if missing
        # Strip whitespace to ensure " dog" and "dog " are treated as duplicates
        pos = data.get("prompt", "").strip()
        neg = data.get("negative_prompt", "").strip()
        
        return (pos, neg)
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

def main():
    if not os.path.isdir(PAYLOADS_DIR):
        print(f"Error: Payloads directory not found at {PAYLOADS_DIR}")
        return

    print(f"Scanning directory: {PAYLOADS_DIR}")
    if DRY_RUN:
        print("--- DRY RUN MODE: No files will be deleted ---")

    # Dictionary to store groups: 
    # Key = (positive_prompt, negative_prompt)
    # Value = List of filenames
    prompt_groups = {}

    files = [f for f in os.listdir(PAYLOADS_DIR) if f.endswith(".json") and f != "payload_latest.json"]
    total_files = len(files)
    print(f"Found {total_files} JSON files. Processing...")

    for filename in files:
        filepath = os.path.join(PAYLOADS_DIR, filename)
        prompts = get_prompts_from_file(filepath)

        if prompts:
            if prompts not in prompt_groups:
                prompt_groups[prompts] = []
            prompt_groups[prompts].append(filename)

    # Process duplicates
    deleted_count = 0
    kept_count = 0

    for prompts, file_list in prompt_groups.items():
        # If there is only 1 file for these prompts, keep it.
        if len(file_list) < 2:
            kept_count += 1
            continue

        # Found duplicates!
        # Sort files by timestamp (Newest last)
        # We want to KEEP the newest one.
        file_list.sort(key=get_file_timestamp)
        
        # The last item is the newest (Keeper)
        keeper = file_list[-1]
        
        # All items before the last one are to be deleted
        to_delete = file_list[:-1]

        print(f"\nDuplicate Group Found:")
        print(f"  Prompt (trunc): {prompts[0][:50]}...")
        print(f"  Keeping newest: {keeper}")
        
        for file_to_remove in to_delete:
            full_path_remove = os.path.join(PAYLOADS_DIR, file_to_remove)
            
            if DRY_RUN:
                print(f"  [DRY RUN] Would delete: {file_to_remove}")
            else:
                try:
                    os.remove(full_path_remove)
                    print(f"  [DELETED] {file_to_remove}")
                    deleted_count += 1
                except OSError as e:
                    print(f"  [ERROR] Could not delete {file_to_remove}: {e}")

    print("-" * 30)
    if DRY_RUN:
        print("Done (DRY RUN). No files were actually deleted.")
        print("Set DRY_RUN = False in the script to execute deletion.")
    else:
        print(f"Done. Deleted {deleted_count} duplicate files.")

if __name__ == "__main__":
    main()