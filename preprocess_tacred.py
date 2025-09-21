import argparse
import json
import os
from tqdm import tqdm
import sys

# Add project root to path for tokenizer
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from pytorch_transformers import RobertaTokenizer

def char_to_word_idx(text, char_offset, tokens):
    """Converts a character offset to a word (token) index."""
    current_char_offset = 0
    for i, token in enumerate(tokens):
        # Handle special Roberta tokens that start with 'Ġ'
        token_text = token.lstrip('Ġ')
        
        # Find the start of the token in the original text
        token_start_char = text.find(token_text, current_char_offset)
        if token_start_char == -1:
            # Fallback for tokens that might not perfectly match (e.g. due to normalization)
            # This is a simple heuristic and might not be perfect.
            current_char_offset += 1
            continue

        token_end_char = token_start_char + len(token_text)
        
        if token_start_char <= char_offset < token_end_char:
            return i
        
        current_char_offset = token_end_char
    return -1 # Return -1 if not found

def preprocess_tacred_data(input_dir, output_dir, tokenizer):
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'dev', 'test']:
        input_path = os.path.join(input_dir, f'{split}.json')
        output_path = os.path.join(output_dir, f'{split}.json')

        if not os.path.exists(input_path):
            print(f"Warning: Input file not found at {input_path}. Skipping.")
            continue

        processed_lines = []
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {input_path}.")
                continue

            for line_obj in tqdm(data, desc=f'Processing {split} data'):
                if 'text' not in line_obj or 'ents' not in line_obj or len(line_obj['ents']) < 2:
                    continue

                text = line_obj['text']
                tokens = tokenizer.tokenize(text)

                # Assume first entity is subject, second is object
                subj_char_start = line_obj['ents'][0][1]
                obj_char_start = line_obj['ents'][1][1]

                subj_word_start = char_to_word_idx(text, subj_char_start, tokens)
                obj_word_start = char_to_word_idx(text, obj_char_start, tokens)
                
                # Heuristic for end index: assume single token entity for simplicity
                # This is a limitation, but required to fit the TREXProcessor format
                subj_word_end = subj_word_start
                obj_word_end = obj_word_start

                if subj_word_start != -1 and obj_word_start != -1:
                    new_record = {
                        'token': tokens,
                        'relation': line_obj.get('relation', line_obj.get('label')), # Use relation or label
                        'subj_start': subj_word_start,
                        'subj_end': subj_word_end,
                        'obj_start': obj_word_start,
                        'obj_end': obj_word_end
                    }
                    processed_lines.append(json.dumps(new_record, ensure_ascii=False))

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
        
        print(f"Processed and saved {len(processed_lines)} lines to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess TACRED data for K-Adapter pre-training.")
    parser.add_argument("--input_dir", type=str, default="./data/tacred", help="Input directory containing TACRED train/dev/test.json.")
    parser.add_argument("--output_dir", type=str, default="./data/tacred_pretrain", help="Output directory for processed JSONL files.")
    args = parser.parse_args()

    print("Loading RobertaTokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    preprocess_tacred_data(args.input_dir, args.output_dir, tokenizer)