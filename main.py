# GLiNER Proof of Concept
# This script demonstrates how to use GLiNER for Named Entity Recognition

from gliner import GLiNER

def main():
    # Load the multilingual model with specific configuration to avoid warnings
    model = GLiNER.from_pretrained(
        "urchade/gliner_multi-v2.1",
        model_max_length=512  # Set a maximum length to avoid truncation warning
    )
    
    # Sample text for demonstration
    text = """
    Cristiano Ronaldo dos Santos Aveiro was born on 5 February 1985 in Portugal.
    He plays for Al Nassr and the Portugal national team.
    Ronaldo has won five Ballon d'Or awards and three UEFA Men's Player of the Year Awards.
    """
    
    # Define entity types we want to extract
    labels = ["person", "award", "date", "organization", "location"]
    
    # Get predictions
    print("Extracting entities from text...")
    entities = model.predict_entities(text, labels)
    
    # Display results
    print("\nExtracted entities:")
    for entity in entities:
        print(f"{entity['text']} => {entity['label']}")

if __name__ == "__main__":
    main()

