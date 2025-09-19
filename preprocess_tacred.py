import argparse
import json
import os
import sys
from tqdm import tqdm

# Add project root to path for tokenizer
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from pytorch_transformers import RobertaTokenizer

def preprocess_tacred_data(input_dir, output_dir, tokenizer):
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'dev', 'test']:
        input_path = os.path.join(input_dir, f'{split}.json')
        output_path = os.path.join(output_dir, f'{split}.json')

        processed_lines = []
        with open(input_path, 'r', encoding='utf-8') as f:
            # TACRED JSON is a list of objects, not JSONL
            data = json.load(f)
            for line_obj in tqdm(data, desc=f'Processing {split} data'):
                # Ensure 'text' key exists
                if 'text' in line_obj:
                    # Tokenize the 'text' field
                    # Using tokenizer.tokenize() which returns a list of strings
                    tokenized_text = tokenizer.tokenize(line_obj['text'])
                    # Add the 'token' field as expected by utils_glue.py
                    line_obj['token'] = tokenized_text

                    # Extract subject and object spans from 'h' and 't'
                    # Extract subject and object spans from 'h' and 't'
                    if 'h' in line_obj and 't' in line_obj and 'relation' in line_obj:
                        subj_start = line_obj['h'][0]
                        subj_end = line_obj['h'][1]
                        obj_start = line_obj['t'][0]
                        obj_end = line_obj['t'][1]

                        # Map 'relation' to 'label'
                        line_obj['label'] = line_obj['relation']

                        # Add these extracted fields to the line_obj for TREXProcessor
                        line_obj['subj_start'] = subj_start
                        line_obj['subj_end'] = subj_end
                        line_obj['obj_start'] = obj_start
                        line_obj['obj_end'] = obj_end

                        processed_lines.append(json.dumps(line_obj, ensure_ascii=False))
                    else:
                        # print(f"Warning: Missing 'h', 't', or 'relation' key in a line. Skipping.")
                        pass # Silently skip lines that don't have the required keys
                else:
                    print(f"Warning: 'text' key not found in a line in {input_path}. Skipping.")

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
        print(f"Processed {len(processed_lines)} lines and saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess TACRED data for K-Adapter pre-training.")
    parser.add_argument("--input_dir", type=str, default="./data/tacred", help="Input directory containing TACRED train/dev/test.json.")
    parser.add_argument("--output_dir", type=str, default="./data/tacred_processed_for_pretrain", help="Output directory for processed JSONL files.")
    args = parser.parse_args()

    print("Loading RobertaTokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    # Ensure pad_token_id is set for older tokenizers
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    preprocess_tacred_data(args.input_dir, args.output_dir, tokenizer)
