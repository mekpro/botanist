import gradio as gr
from unsloth import FastModel
import torch
from PIL import Image
import base64
from io import BytesIO
import warnings
import json

model, tokenizer = FastModel.from_pretrained(
#    model_name="mekpro/gemma-3n-botanist-grpo6p-merged",
    model_name="mekpro/gemma-3n-botanist8-merged",
    dtype=torch.bfloat16,
    max_seq_length=512,
    load_in_4bit=False,  # Disable 4-bit to avoid type casting errors
)
FastModel.for_inference(model) 
torch._dynamo.config.cache_size_limit = 256  # Default is 64, increase as needed

botanist_prompt="As a botanist, observe the image of the flower and describe its visual features and species_name in JSON format. {color, inflorescencetype, inflorescence_description, flower_arrangement, flower_density, species, family, genus}"

def safe_load_image(image):
    """Safely load image, handling EXIF errors"""
    if image is None:
        return None
    
    try:
        # Try to apply EXIF transpose
        from PIL import ImageOps
        return ImageOps.exif_transpose(image)
    except Exception as e:
        # If EXIF transpose fails, return original image
        warnings.warn(f"Failed to apply EXIF transpose: {e}")
        return image

def validate_json_output(output):
    """Validate if the output is valid JSON"""
    try:
        # Try to parse the JSON
        json_data = json.loads(output)
        # Check if required fields are present
        required_fields = ["color", "inflorescencetype", "inflorescence_description", 
                         "flower_arrangement", "flower_density", "species", "family", "genus"]
        
        # Allow partial matches or variations in field names
        found_fields = [field.lower() for field in json_data.keys()]
        
        # Just check if we have some JSON structure, not strict field validation
        if isinstance(json_data, dict) and len(json_data) > 0:
            return True, json_data
        else:
            return False, None
    except json.JSONDecodeError:
        # Try to extract JSON from the output if it contains other text
        import re
        json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                if isinstance(json_data, dict) and len(json_data) > 0:
                    return True, json_data
            except:
                pass
        return False, None

def process_input(image):
    """Process image input for botanist analysis with JSON validation and retry"""
    
    if not image:
        return "Please provide an image."
    
    # Safely load image with EXIF handling
    image = safe_load_image(image)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use multimodal format with botanist prompt
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": botanist_prompt}
                ]
            }]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")
            
        except Exception as e:
            # Fallback to text-only if multimodal fails
            print(f"Multimodal format failed: {e}")
            prompt = f"[Image provided] {botanist_prompt}"
            
            # Fix: Ensure content is a list format for consistency
            messages = [{
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }]
            
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
            except Exception as e2:
                # Final fallback: use simple string content
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
        
        # Generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
        )
        
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Validate JSON output
        is_valid, json_data = validate_json_output(output)
        
        if is_valid:
            # Return formatted JSON
            return json.dumps(json_data, indent=2)
        else:
            print(f"Attempt {attempt + 1}/{max_retries}: Invalid JSON output, retrying...")
            if attempt == max_retries - 1:
                # On last attempt, return the raw output with a note
                return f"Warning: Could not generate valid JSON after {max_retries} attempts.\n\nRaw output:\n{output}"
    
    return "Error: Failed to generate output after all retries."

# Simple interface
demo = gr.Interface(
    fn=process_input,
    inputs=gr.Image(type="pil", label="Upload Flower Image"),
    outputs=gr.Textbox(lines=10, label="Botanical Analysis"),
    title="Gemma 3n Botanist by @mekpro",
    description="Upload flower image and we will tell the species!",
    flagging_mode="never"  # Disable flag button
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
