# GLiNER Proof of Concept
# This script demonstrates how to use GLiNER for Named Entity Recognition

import json
import argparse
from gliner import GLiNER

def load_examples(file_path):
    """Load examples from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['examples'], data['entity_types']

def process_example(model, example, entity_types):
    """Process a single example and extract entities."""
    print(f"\n--- Example {example['id']}: {example['description']} ---")
    print(f"Text: {example['text'][:100]}..." if len(example['text']) > 100 else f"Text: {example['text']}")
    
    # Get predictions
    entities = model.predict_entities(example['text'], entity_types)
    
    # Display results
    print("Extracted entities:")
    for entity in entities:
        print(f"  {entity['text']} => {entity['label']}")
    
    return entities

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GLiNER Named Entity Recognition')
    parser.add_argument('--example', type=int, help='ID of the specific example to process')
    parser.add_argument('--file', type=str, default='examples.json', help='Path to examples JSON file')
    args = parser.parse_args()
    
    # Load the multilingual model with specific configuration to avoid warnings
    print("Loading GLiNER model...")
    model = GLiNER.from_pretrained(
        "urchade/gliner_multi-v2.1",
        model_max_length=512  # Set a maximum length to avoid truncation warning
    )
    
    # Load examples from JSON file
    examples, entity_types = load_examples(args.file)
    print(f"Loaded {len(examples)} examples with {len(entity_types)} entity types")
    
    # Process examples
    if args.example:
        # Process only the specified example
        example = next((ex for ex in examples if ex['id'] == args.example), None)
        if example:
            process_example(model, example, entity_types)
        else:
            print(f"Example with ID {args.example} not found")
    else:
        # Process all examples
        all_entities = []
        for example in examples:
            entities = process_example(model, example, entity_types)
            all_entities.extend(entities)
        
        # Print summary
        print("\n--- Summary ---")
        print(f"Processed {len(examples)} examples")
        print(f"Total entities found: {len(all_entities)}")
        
        # Count entities by type
        entity_counts = {}
        for entity in all_entities:
            entity_type = entity['label']
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        print("Entities by type:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  {entity_type}: {count}")

if __name__ == "__main__":
    main()

