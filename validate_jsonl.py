import json
import sys

REQUIRED_FIELDS = [
    "observation_id",
    "species", 
    "family",
    "genus",
    "color",
    "inflorescencetype",
    "inflorescence_description",
    "flower_arrangement",
    "flower_density",
    "unique_visual_description",
    "morphological_traits_observable_in_photograph",
    "visual_contrast_with_similar_species"
]

def validate_observation(obs):
    """
    Validate a single observation.
    Returns None if valid, error message if invalid.
    """
    # Check all required fields are present
    for field in REQUIRED_FIELDS:
        if field not in obs:
            return f"Missing required field: {field}"
        if not isinstance(obs[field], str):
            return f"Field {field} must be a string, got {type(obs[field])}"
        if not obs[field].strip():
            return f"Field {field} is empty"
    
    return None

def extract_valid_json_objects(text):
    # Filter out lines starting with ```
    lines = text.splitlines()
    filtered_lines = [line for line in lines if not line.strip().startswith('```')]
    text = '\n'.join(filtered_lines) + '\n'

    objects = []
    pos = 0
    decoder = json.JSONDecoder()
    while pos < len(text):
        try:
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break

            # If doesn't start with {, skip to next newline
            if text[pos] != '{':
                next_nl = text.find('\n', pos)
                if next_nl == -1:
                    break
                pos = next_nl + 1
                continue

            # Parse the JSON object
            obj, end = decoder.raw_decode(text, pos)
            objects.append(obj)
            pos = end
        except json.JSONDecodeError:
            # If parsing fails, skip to the next potential object start
            next_start = text.find('{', pos + 1)
            if next_start == -1:
                break
            pos = next_start
    return objects

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f:
        input_text = f.read()

    parsed_objects = extract_valid_json_objects(input_text)

    valid_objects = []
    for obj in parsed_objects:
        error = validate_observation(obj)
        if error is None:
            valid_objects.append(obj)

    with open(output_file, 'w') as out:
        for obj in valid_objects:
            out.write(json.dumps(obj) + '\n')