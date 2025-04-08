# GLiNER Proof of Concept
# This script demonstrates how to use GLiNER for Named Entity Recognition

import argparse
import json

from gliner import GLiNER


def load_examples(file_path):
    """Load examples from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["examples"], data["entity_types"]


def process_long_text(model, text, entity_types, chunk_size=512, overlap=100):
    """Process a long text by splitting it into overlapping chunks."""
    # For shorter texts, process directly
    if len(text) <= chunk_size:
        return model.predict_entities(text, entity_types)

    # Split text into chunks with overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end

    # Process each chunk
    all_entities = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_entities = model.predict_entities(chunk, entity_types)

        # Adjust entity positions for chunks after the first one
        if i > 0:
            offset = chunks[i - 1].rfind(" ", 0, len(chunks[i - 1]) - overlap)
            if offset == -1:
                offset = len(chunks[i - 1]) - overlap

            # Only add entities that aren't duplicates from the overlap
            for entity in chunk_entities:
                # Check if this entity is a duplicate from the previous chunk's overlap
                is_duplicate = False
                for prev_entity in all_entities:
                    if (
                        entity["text"] == prev_entity["text"]
                        and entity["label"] == prev_entity["label"]
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    all_entities.append(entity)
        else:
            all_entities.extend(chunk_entities)

    return all_entities


def process_example(model, example, entity_types):
    """Process a single example and extract entities."""
    print(f"\n--- Example {example['id']}: {example['description']} ---")
    print(
        f"Text: {example['text'][:100]}..."
        if len(example["text"]) > 100
        else f"Text: {example['text']}"
    )

    # Get predictions, handling long texts appropriately
    entities = process_long_text(model, example["text"], entity_types)

    # Display results
    print("Extracted entities:")
    for entity in entities:
        print(f"  {entity['text']} => {entity['label']}")

    return entities


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GLiNER Named Entity Recognition")
    parser.add_argument(
        "--example", type=int, help="ID of the specific example to process"
    )
    parser.add_argument(
        "--file", type=str, default="examples.json", help="Path to examples JSON file"
    )
    args = parser.parse_args()

    # Load the multilingual model
    print("Loading GLiNER model...")
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

    # Load examples from JSON file
    examples, entity_types = load_examples(args.file)
    print(f"Loaded {len(examples)} examples with {len(entity_types)} entity types")

    # Process examples
    if args.example:
        # Process only the specified example
        example = next((ex for ex in examples if ex["id"] == args.example), None)
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
            entity_type = entity["label"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        print("Entities by type:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  {entity_type}: {count}")


if __name__ == "__main__":
    main()
