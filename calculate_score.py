#!/usr/bin/env python3
import re
import sys
from typing import List, Dict, Tuple

def parse_results_file(filename: str) -> List[Dict]:
    """Parse the results file and extract image data."""
    images = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by image sections
    image_sections = content.split('\nImage ')[1:]  # Skip the header part
    
    for section in image_sections:
        lines = section.strip().split('\n')
        if len(lines) < 3:
            continue
            
        # Extract image info
        image_data = {
            'predictions': []
        }
        
        # Parse true values
        for line in lines:
            if 'True species:' in line:
                match = re.search(r'True species: (.*?) \| True inflorescence: (.*)', line)
                if match:
                    image_data['true_species'] = match.group(1).strip()
                    image_data['true_inflorescence'] = match.group(2).strip()
            elif line.strip().startswith('Prediction'):
                # Parse prediction
                pred_match = re.search(r'Species: (.*?) \| Inflorescence: (.*)', line)
                if pred_match:
                    image_data['predictions'].append({
                        'species': pred_match.group(1).strip(),
                        'inflorescence': pred_match.group(2).strip()
                    })
                elif 'Invalid JSON response' not in line:
                    # Sometimes format might be different, try to extract what we can
                    pass
        
        if 'true_species' in image_data and 'true_inflorescence' in image_data:
            images.append(image_data)
    
    return images

def get_genus(species: str) -> str:
    """Extract genus (first word) from species name."""
    return species.split()[0] if species else ""

def check_species_match(true_species: str, pred_species: str) -> Tuple[bool, bool]:
    """
    Check if species match.
    Returns: (full_match, genus_match)
    """
    # Clean up the species names
    true_species = true_species.strip()
    pred_species = pred_species.strip()
    
    # Full match
    if true_species.lower() == pred_species.lower():
        return True, True
    
    # Genus match
    true_genus = get_genus(true_species)
    pred_genus = get_genus(pred_species)
    
    if true_genus and pred_genus and true_genus.lower() == pred_genus.lower():
        return False, True
    
    return False, False

def check_inflorescence_match(true_inflor: str, pred_inflor: str) -> Tuple[bool, bool]:
    """
    Check if inflorescence types match.
    Returns: (exact_match, substring_match)
    """
    # Clean up and lowercase for comparison
    true_inflor = true_inflor.strip().lower()
    pred_inflor = pred_inflor.strip().lower()
    
    # Exact match
    if true_inflor == pred_inflor:
        return True, False
    
    # Substring match - check if one is substring of the other
    if true_inflor in pred_inflor or pred_inflor in true_inflor:
        return False, True
    
    return False, False

def calculate_accuracy(images: List[Dict], debug=False) -> Dict:
    """Calculate accuracy metrics for all images."""
    total = len(images)
    
    # Top-1 metrics
    top1_species_full = 0
    top1_species_genus = 0
    top1_inflor_exact = 0
    top1_inflor_substring = 0
    
    # Top-5 metrics
    top5_species_full = 0
    top5_species_genus_only = 0  # Genus match but NOT full match
    top5_inflor_exact = 0
    top5_inflor_substring_only = 0  # Substring match but NOT exact match
    
    for image in images:
        true_species = image['true_species']
        true_inflor = image['true_inflorescence']
        predictions = image['predictions']
        
        # First pass: check all predictions to determine best match type
        has_species_full = False
        has_species_genus = False
        has_inflor_exact = False
        has_inflor_substring = False
        
        for i, pred in enumerate(predictions[:5]):  # Only consider top 5
            pred_species = pred['species']
            pred_inflor = pred['inflorescence']
            
            # Check species match
            full_match, genus_match = check_species_match(true_species, pred_species)
            
            # Check inflorescence match
            inflor_exact, inflor_substring = check_inflorescence_match(true_inflor, pred_inflor)
            
            # Top-1 (first prediction only)
            if i == 0:
                if full_match:
                    top1_species_full += 1
                elif genus_match:
                    top1_species_genus += 1
                
                if inflor_exact:
                    top1_inflor_exact += 1
                elif inflor_substring:
                    top1_inflor_substring += 1
            
            # Track what matches we found in top-5
            if full_match:
                has_species_full = True
            elif genus_match:
                has_species_genus = True
            
            if inflor_exact:
                has_inflor_exact = True
            elif inflor_substring:
                has_inflor_substring = True
        
        # Now count for top-5 based on best match found
        if has_species_full:
            top5_species_full += 1
        elif has_species_genus:
            top5_species_genus_only += 1
        
        if has_inflor_exact:
            top5_inflor_exact += 1
        elif has_inflor_substring:
            top5_inflor_substring_only += 1
    
    return {
        'total': total,
        'top1': {
            'species_full': top1_species_full,
            'species_genus': top1_species_genus,
            'inflor_exact': top1_inflor_exact,
            'inflor_substring': top1_inflor_substring
        },
        'top5': {
            'species_full': top5_species_full,
            'species_genus': top5_species_genus_only,
            'inflor_exact': top5_inflor_exact,
            'inflor_substring': top5_inflor_substring_only
        }
    }

def print_results(metrics: Dict):
    """Print results in the requested format."""
    total = metrics['total']
    
    print("----------------------------------------")
    print("TOP-1 ACCURACY (First prediction only)")
    print("----------------------------------------")
    print()
    
    # Species Top-1
    print("Species Top-1 accuracy:")
    top1_species_full = metrics['top1']['species_full']
    top1_species_genus = metrics['top1']['species_genus']
    print(f"Full correct: {top1_species_full}/{total}")
    print(f"Full accuracy: {top1_species_full/total*100:.2f}%")
    print(f"Half correct (genus only): {top1_species_genus}/{total}")
    print(f"Half accuracy: {top1_species_genus/total*100:.2f}%")
    total_genus_correct = top1_species_full + top1_species_genus
    print(f"Total with at least genus correct: {total_genus_correct}/{total}")
    print(f"Combined accuracy (full + half): {total_genus_correct/total*100:.2f}%")
    print()
    
    # Inflorescence Top-1
    print("Inflorescence type Top-1 accuracy:")
    top1_inflor_exact = metrics['top1']['inflor_exact']
    top1_inflor_substring = metrics['top1']['inflor_substring']
    print(f"Exact match: {top1_inflor_exact}/{total}")
    print(f"Exact match accuracy: {top1_inflor_exact/total*100:.2f}%")
    print(f"Substring match: {top1_inflor_substring}/{total}")
    print(f"Substring match accuracy: {top1_inflor_substring/total*100:.2f}%")
    total_inflor_match = top1_inflor_exact + top1_inflor_substring
    print(f"Total matches (exact + substring): {total_inflor_match}/{total}")
    print(f"Combined accuracy: {total_inflor_match/total*100:.2f}%")
    
    print()
    print("----------------------------------------")
    print("TOP-5 ACCURACY (Any of 5 predictions)")
    print("----------------------------------------")
    print()
    
    # Species Top-5
    print("Species Top-5 accuracy:")
    top5_species_full = metrics['top5']['species_full']
    top5_species_genus = metrics['top5']['species_genus']
    print(f"Full correct: {top5_species_full}/{total}")
    print(f"Full accuracy: {top5_species_full/total*100:.2f}%")
    print(f"Half correct (genus only): {top5_species_genus}/{total}")
    print(f"Half accuracy: {top5_species_genus/total*100:.2f}%")
    total_genus_correct_5 = top5_species_full + top5_species_genus
    print(f"Total with at least genus correct: {total_genus_correct_5}/{total}")
    print(f"Combined accuracy (full + half): {total_genus_correct_5/total*100:.2f}%")
    print()
    
    # Inflorescence Top-5
    print("Inflorescence type Top-5 accuracy:")
    top5_inflor_exact = metrics['top5']['inflor_exact']
    top5_inflor_substring = metrics['top5']['inflor_substring']
    print(f"Exact match: {top5_inflor_exact}/{total}")
    print(f"Exact match accuracy: {top5_inflor_exact/total*100:.2f}%")
    print(f"Substring match: {top5_inflor_substring}/{total}")
    print(f"Substring match accuracy: {top5_inflor_substring/total*100:.2f}%")
    total_inflor_match_5 = top5_inflor_exact + top5_inflor_substring
    print(f"Total matches (exact + substring): {total_inflor_match_5}/{total}")
    print(f"Combined accuracy: {total_inflor_match_5/total*100:.2f}%")

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python calculate_score.py <results_file>")
        print("Example: python calculate_score.py results_grpo.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Parse results file
    try:
        images = parse_results_file(filename)
        print(f"Parsed {len(images)} images from {filename}")
        print()
        
        # Calculate metrics
        metrics = calculate_accuracy(images)
        
        # Print results
        print_results(metrics)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()