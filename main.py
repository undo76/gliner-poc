# GLiNER Proof of Concept
# This script demonstrates how to use GLiNER for Named Entity Recognition

import argparse
import json
import re
from collections import defaultdict

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


def highlight_entities_in_text(text, entities):
    """Highlight entities in text using ANSI color codes."""
    # Define colors for different entity types
    colors = {
        "person": "\033[1;31m",      # Bold Red
        "organization": "\033[1;34m", # Bold Blue
        "location": "\033[1;32m",     # Bold Green
        "date": "\033[1;33m",         # Bold Yellow
        "product": "\033[1;35m",      # Bold Magenta
        "event": "\033[1;36m",        # Bold Cyan
        "award": "\033[1;95m",        # Bold Light Magenta
    }
    reset = "\033[0m"  # Reset color
    
    # Sort entities by their position in the text (to handle overlapping entities)
    # We'll use a defaultdict to store entities by their starting positions
    positions = defaultdict(list)
    
    # Find all occurrences of each entity in the text
    for entity in entities:
        entity_text = entity["text"]
        entity_type = entity["label"]
        
        # Find all occurrences of this entity in the text
        for match in re.finditer(re.escape(entity_text), text):
            start, end = match.span()
            positions[start].append((end, entity_text, entity_type))
    
    # Build the highlighted text
    result = []
    last_end = 0
    
    # Sort positions to process them in order
    for start in sorted(positions.keys()):
        # Add text before this entity
        if start > last_end:
            result.append(text[last_end:start])
        
        # Get the entity with the longest span at this position
        entities_at_pos = sorted(positions[start], key=lambda x: x[0], reverse=True)
        end, entity_text, entity_type = entities_at_pos[0]
        
        # Add the highlighted entity
        color = colors.get(entity_type, "\033[1m")  # Default to bold if type not found
        result.append(f"{color}{text[start:end]}{reset}")
        
        last_end = end
    
    # Add any remaining text
    if last_end < len(text):
        result.append(text[last_end:])
    
    return "".join(result)


def process_example(model, example, entity_types):
    """Process a single example and extract entities."""
    print(f"\n--- Example {example['id']}: {example['description']} ---")
    
    # Get predictions, handling long texts appropriately
    entities = process_long_text(model, example["text"], entity_types)
    
    # Display highlighted text
    highlighted_text = highlight_entities_in_text(example["text"], entities)
    print("\nHighlighted Text:")
    print(highlighted_text)
    
    # Display entity list
    print("\nExtracted entities:")
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

        # Display color legend
        colors = {
            "person": "\033[1;31m",      # Bold Red
            "organization": "\033[1;34m", # Bold Blue
            "location": "\033[1;32m",     # Bold Green
            "date": "\033[1;33m",         # Bold Yellow
            "product": "\033[1;35m",      # Bold Magenta
            "event": "\033[1;36m",        # Bold Cyan
            "award": "\033[1;95m",        # Bold Light Magenta
        }
        reset = "\033[0m"  # Reset color
        
        print("\nColor Legend:")
        for entity_type in sorted(entity_counts.keys()):
            color = colors.get(entity_type, "\033[1m")
            print(f"  {color}{entity_type}{reset}")
        
        print("\nEntities by type:")
        for entity_type, count in sorted(entity_counts.items()):
            color = colors.get(entity_type, "\033[1m")
            print(f"  {color}{entity_type}{reset}: {count}")


if __name__ == "__main__":
    main()
