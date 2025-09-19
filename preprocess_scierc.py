import json
from datasets import load_dataset

def process_scierc_split(split_data):
    """
    Processes a split of the SciERC dataset to extract relation triples.
    """
    triples = []
    for entry in split_data:
        entities = entry['entities']
        relations = entry['relations']

        # Create a quick lookup for entities by their ID
        entity_map = {entity['id']: entity['text'] for entity in entities}

        for rel in relations:
            # Ensure both head and tail entities are in our map
            if rel['head'] in entity_map and rel['tail'] in entity_map:
                head_entity = entity_map[rel['head']]
                tail_entity = entity_map[rel['tail']]
                relation_label = rel['label']
                
                # Format: [Subject, Relation, Object]
                triples.append([head_entity, relation_label, tail_entity])
    return triples

def main():
    """
    Main function to download, process, and save the SciERC dataset.
    """
    print("Loading SciERC dataset from Hugging Face...")
    # Load the dataset from Hugging Face
    scierc_dataset = load_dataset("hrithikpiyush/scierc")
    print("Dataset loaded successfully.")

    # Process each split
    print("Processing train split...")
    train_triples = process_scierc_split(scierc_dataset['train'])
    
    print("Processing validation split...")
    dev_triples = process_scierc_split(scierc_dataset['validation'])
    
    print("Processing test split...")
    test_triples = process_scierc_split(scierc_dataset['test'])

    # Save the processed data to JSON files in a new directory
    output_dir = "./data/scierc_processed"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving processed train split to {output_dir}/train.json...")
    with open(f"{output_dir}/train.json", "w") as f:
        json.dump(train_triples, f, indent=4)

    print(f"Saving processed validation split to {output_dir}/dev.json...")
    with open(f"{output_dir}/dev.json", "w") as f:
        json.dump(dev_triples, f, indent=4)

    print(f"Saving processed test split to {output_dir}/test.json...")
    with open(f"{output_dir}/test.json", "w") as f:
        json.dump(test_triples, f, indent=4)
        
    print("\nPreprocessing finished!")
    print(f"Processed data saved in '{output_dir}' directory.")
    print(f"Found {len(train_triples)} triples in train set.")
    print(f"Found {len(dev_triples)} triples in validation set.")
    print(f"Found {len(test_triples)} triples in test set.")

if __name__ == "__main__":
    import os
    main()
