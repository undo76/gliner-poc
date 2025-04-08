# GLiNER Proof of Concept
# This script demonstrates how to use GLiNER for Named Entity Recognition

import argparse
import json
import re
from collections import defaultdict
import warnings

from gliner import GLiNER
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table


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
    """Highlight entities in text using Rich library."""
    # Available rich styles for highlighting
    styles = [
        "bold red",
        "bold blue",
        "bold green",
        "bold yellow",
        "bold magenta",
        "bold cyan",
        "bold purple",
        "bold white on red",
        "bold white on blue",
        "bold white on green",
        "bold white on magenta",
    ]

    # Create a console for rich output
    console = Console()

    # Create a mapping of entity types to styles
    entity_types = set(entity["label"] for entity in entities)
    entity_styles = {}

    # Assign a random style to each entity type
    for entity_type in entity_types:
        # Use modulo to cycle through styles if we have more entity types than styles
        entity_styles[entity_type] = styles[len(entity_styles) % len(styles)]

    # Sort entities by their position in the text (to handle overlapping entities)
    positions = defaultdict(list)

    # Find all occurrences of each entity in the text
    for entity in entities:
        entity_text = entity["text"]
        entity_type = entity["label"]

        # Find all occurrences of this entity in the text
        for match in re.finditer(re.escape(entity_text), text):
            start, end = match.span()
            positions[start].append((end, entity_text, entity_type))

    # Build the highlighted text using Rich
    rich_text = Text()
    last_end = 0

    # Sort positions to process them in order
    for start in sorted(positions.keys()):
        # Add text before this entity
        if start > last_end:
            rich_text.append(text[last_end:start])

        # Get the entity with the longest span at this position
        entities_at_pos = sorted(positions[start], key=lambda x: x[0], reverse=True)
        end, entity_text, entity_type = entities_at_pos[0]

        # Add the highlighted entity
        style = entity_styles.get(entity_type, "bold")
        rich_text.append(text[start:end], style=style)

        last_end = end

    # Add any remaining text
    if last_end < len(text):
        rich_text.append(text[last_end:])

    # Create a panel with the highlighted text
    panel = Panel(rich_text, title="Highlighted Text", expand=False)

    # Create a legend table
    table = Table(title="Entity Types Legend")
    table.add_column("Entity Type")
    table.add_column("Style")

    for entity_type, style in entity_styles.items():
        table.add_row(entity_type, Text(entity_type, style=style))

    return console, panel, table, entity_styles


def process_example(model, example, entity_types):
    """Process a single example and extract entities."""
    console = Console()
    console.print(f"\n[bold]Example {example['id']}: {example['description']}[/bold]")

    # Get predictions, handling long texts appropriately
    entities = process_long_text(model, example["text"], entity_types)

    # Display highlighted text using Rich
    console, panel, legend, entity_styles = highlight_entities_in_text(
        example["text"], entities
    )

    # Print the panel with highlighted text
    console.print(panel)

    # Print the legend
    console.print(legend)

    # Display entity list with rich formatting
    console.print("\n[bold]Extracted entities:[/bold]")

    # Group entities by type for better organization
    entities_by_type = defaultdict(list)
    for entity in entities:
        entities_by_type[entity["label"]].append(entity["text"])

    # Create a table for entities
    entity_table = Table(show_header=True)
    entity_table.add_column("Entity Type")
    entity_table.add_column("Entities")

    for entity_type, entity_texts in sorted(entities_by_type.items()):
        style = entity_styles.get(entity_type, "bold")
        entity_table.add_row(
            Text(entity_type, style=style), Text(", ".join(entity_texts))
        )

    console.print(entity_table)

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

    # Suppress specific warning about sentencepiece tokenizer
    warnings.filterwarnings("ignore", message="The sentencepiece tokenizer.*")

    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1", max_length=512)

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

        # Print summary with Rich
        console = Console()
        console.print("\n[bold]Summary[/bold]")
        console.print(f"Processed {len(examples)} examples")
        console.print(f"Total entities found: {len(all_entities)}")

        # Count entities by type
        entity_counts = {}
        for entity in all_entities:
            entity_type = entity["label"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        # Create a summary table
        summary_table = Table(title="Entities by Type")
        summary_table.add_column("Entity Type")
        summary_table.add_column("Count")
        summary_table.add_column("Percentage", justify="right")

        # Available rich styles for highlighting
        styles = [
            "bold red",
            "bold blue",
            "bold green",
            "bold yellow",
            "bold magenta",
            "bold cyan",
            "bold purple",
            "bold white on red",
            "bold white on blue",
            "bold white on green",
            "bold white on magenta",
        ]

        # Assign styles to entity types
        entity_styles = {}
        for i, entity_type in enumerate(sorted(entity_counts.keys())):
            entity_styles[entity_type] = styles[i % len(styles)]

        # Calculate total for percentage
        total_entities = len(all_entities)

        # Add rows to the table
        for entity_type, count in sorted(
            entity_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total_entities) * 100
            summary_table.add_row(
                Text(entity_type, style=entity_styles[entity_type]),
                str(count),
                f"{percentage:.1f}%",
            )

        console.print(summary_table)


if __name__ == "__main__":
    main()
