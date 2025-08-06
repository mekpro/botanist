#!/Users/mekpro/github/PlantNet-300K/botan/venv/bin/python3
"""
Dataset preparation script for PlantNet-300K botanist observations.
Creates a HuggingFace dataset mapping plant images to detailed botanical observations.
"""

import os
import json
import csv
import argparse
import asyncio
import aiohttp
import random
import base64
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Value, Image as DatasetImage


class PlantObservationDatasetBuilder:
    def __init__(self, photos_per_species: int = 10, max_concurrent_requests: int = 5, 
                 observations_file: str = "json_observation_list.jsonl"):
        self.photos_per_species = photos_per_species
        self.max_concurrent_requests = max_concurrent_requests
        self.hf_token = os.environ.get("HF_TOKEN")
        self.together_api_key = os.environ.get("TOGETHER_API_KEY")
        
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        if not self.together_api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        
        self.observations_file = observations_file
        self.plant_files_csv = "/Users/mekpro/github/PlantNet-300K/botan/plant_files.csv"
        self.images_root = "/Users/mekpro/github/PlantNet-300K/plantnet_300K/images"
        
        self.observations_by_species = defaultdict(list)
        self.species_to_images = defaultdict(list)
        
        # Progress tracking
        self.total_api_requests = 0
        self.successful_api_requests = 0
        self.failed_api_requests = 0
        self.images_processed = 0
        self.images_with_consensus = 0
        self.images_with_random_selection = 0
        self.species_processed = 0
        self.total_species = 0
        self.start_time = time.time()
        
        # Rate limiting
        self.api_semaphore = None  # Will be initialized in async context
        
        # CSV output file
        self.csv_output_file = None  # Will be set via command line
        
        # Track processed images for resumability
        self.processed_images = set()
        
    def get_dataset_features(self):
        """Get dataset features definition (reusable across methods)"""
        return Features({
            "image": DatasetImage(),
            "observation_id": Value("string"),
            "species": Value("string"),
            "family": Value("string"),
            "genus": Value("string"),
            "color": Value("string"),
            "inflorescencetype": Value("string"),
            "inflorescence_description": Value("string"),
            "flower_arrangement": Value("string"),
            "flower_density": Value("string"),
            "unique_visual_description": Value("string"),
            "morphological_traits_observable_in_photograph": Value("string"),
            "visual_contrast_with_similar_species": Value("string")
        })
    
    def create_dataset_row(self, image, observation):
        """Create a dataset row from image and observation (reusable)"""
        return {
            "image": image,
            "observation_id": str(observation.get("observation_id", "")),
            "species": str(observation.get("species", "")),
            "family": str(observation.get("family", "")),
            "genus": str(observation.get("genus", "")),
            "color": str(observation.get("color", "")),
            "inflorescencetype": str(observation.get("inflorescencetype", "")),
            "inflorescence_description": str(observation.get("inflorescence_description", "")),
            "flower_arrangement": str(observation.get("flower_arrangement", "")),
            "flower_density": str(observation.get("flower_density", "")),
            "unique_visual_description": str(observation.get("unique_visual_description", "")),
            "morphological_traits_observable_in_photograph": str(observation.get("morphological_traits_observable_in_photograph", "")),
            "visual_contrast_with_similar_species": str(observation.get("visual_contrast_with_similar_species", ""))
        }
    
    def print_progress_summary(self, batch_num, total_batches, rows_created=None):
        """Print progress summary (reusable across methods)"""
        elapsed = time.time() - self.start_time
        rate = self.total_api_requests / elapsed if elapsed > 0 else 0
        consensus_rate = (self.images_with_consensus / self.images_processed * 100) if self.images_processed > 0 else 0
        success_rate = (self.successful_api_requests / self.total_api_requests * 100) if self.total_api_requests > 0 else 0
        
        print(f"\n📊 Progress Summary - Batch {batch_num}/{total_batches}")
        print(f"   Species processed: {self.species_processed}/{self.total_species} ({self.species_processed/self.total_species*100:.1f}%) - {self.total_species - self.species_processed} remaining")
        if rows_created is not None:
            print(f"   Total rows created: {rows_created}")
        print(f"   Images processed: {self.images_processed}")
        print(f"   Images with consensus: {self.images_with_consensus} ({consensus_rate:.1f}%)")
        print(f"   Images with random selection: {self.images_with_random_selection}")
        print(f"   API requests: {self.total_api_requests} total, {self.successful_api_requests} successful ({success_rate:.1f}%)")
        print(f"   Request rate: {rate:.1f} req/s")
        print(f"   Time elapsed: {elapsed:.1f}s")
        
    def load_processed_images(self):
        """Load already processed images from CSV file"""
        if self.csv_output_file and os.path.exists(self.csv_output_file):
            print(f"Loading existing results from {self.csv_output_file}...")
            try:
                with open(self.csv_output_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 1:
                            self.processed_images.add(row[0])  # Add image path
                print(f"Found {len(self.processed_images)} already processed images")
            except Exception as e:
                print(f"Error loading existing CSV: {e}")
                self.processed_images = set()
        else:
            print("No existing CSV file found, starting fresh")
            self.processed_images = set()
        
    def save_observation_to_csv(self, image_path: str, observation: Dict):
        """Save a single observation result to CSV file"""
        with open(self.csv_output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write as: filepath,json_observation
            writer.writerow([image_path, json.dumps(observation)])
    
    def load_observations_from_csv(self, csv_file: str) -> List[Tuple[str, Dict]]:
        """Load observations from CSV file"""
        observations = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    image_path, observation_json = row
                    try:
                        observation = json.loads(observation_json)
                        observations.append((image_path, observation))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON for {image_path}: {e}")
                        continue
        return observations
        
    def load_observations(self):
        """Load observations from JSONL file"""
        print("Loading observations...")
        with open(self.observations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    species = obs['species']
                    self.observations_by_species[species].append(obs)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    print(f"Line content: {line[:100]}...")
                    continue
        
        print(f"Loaded observations for {len(self.observations_by_species)} species")
        
    def load_plant_files(self):
        """Load plant files CSV and create species to image mapping"""
        print("Loading plant files...")
        with open(self.plant_files_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    species_with_author, image_path = row
                    # Try to match species name from observations
                    for obs_species in self.observations_by_species.keys():
                        # Convert observation species to underscore format
                        obs_species_underscore = obs_species.replace(' ', '_')
                        if species_with_author.startswith(obs_species_underscore):
                            self.species_to_images[obs_species].append(image_path)
                            break
        
        print(f"Found images for {len(self.species_to_images)} species")
        
    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """Preprocess image: resize/crop to 384x384 and convert to JPEG
        
        Returns:
            Optional[Image.Image]: Preprocessed image or None if error
        """
        try:
            full_path = os.path.join(self.images_root, image_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                print(f"⚠️  Image file not found: {full_path}")
                return None
            
            # Open image
            img = Image.open(full_path)
            
            # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get current dimensions
            width, height = img.size
            
            # Skip if image is too small
            if width < 10 or height < 10:
                print(f"⚠️  Image too small ({width}x{height}): {image_path}")
                return None
            
            # Calculate crop dimensions for center crop to square
            if width > height:
                # Wider image - crop horizontally
                crop_size = height
                left = (width - crop_size) // 2
                top = 0
                right = left + crop_size
                bottom = crop_size
            else:
                # Taller image - crop vertically
                crop_size = width
                left = 0
                top = (height - crop_size) // 2
                right = crop_size
                bottom = top + crop_size
            
            # Crop to square
            img = img.crop((left, top, right, bottom))
            
            # Resize to 384x384
            img = img.resize((384, 384), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"⚠️  Error preprocessing image {image_path}: {e}")
            return None
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode preprocessed image to base64 string
        
        Returns:
            Optional[str]: Base64 encoded image or None if error
        """
        # Preprocess image first
        img = self.preprocess_image(image_path)
        
        if img is None:
            return None
        
        try:
            # Convert to JPEG with quality 80
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            buffer.seek(0)
            
            # Encode to base64
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"⚠️  Error encoding image {image_path}: {e}")
            return None
    
    async def query_together_ai(self, session: aiohttp.ClientSession, image_path: str, 
                               observations: List[Dict], attempt: int) -> Optional[str]:
        """Query Together AI API with image and observations
        
        Note: The semaphore limits concurrent image processing, not individual API calls.
        Each image makes 5 parallel calls to this method.
        """
        async with self.api_semaphore:  # Limit concurrent API calls
            self.total_api_requests += 1
            try:
                # Encode image
                image_base64 = self.encode_image_to_base64(image_path)
                
                if image_base64 is None:
                    self.failed_api_requests += 1
                    return None
                
                # Randomize observation order
                shuffled_obs = observations.copy()
                random.shuffle(shuffled_obs)
                
                # Build prompt
                obs_text = "\n".join([json.dumps(obs) for obs in shuffled_obs])
                
                prompt = f"""Observe the photo and select "observation_id" that is the most close description to this photo. Reply with ONLY the observation_id, nothing else.

{obs_text}"""
                
                # Prepare API request
                headers = {
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "google/gemma-3n-E4B-it",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1
                }
                
                async with session.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data['choices'][0]['message']['content'].strip()
                        
                        # Extract observation_id from response
                        # Try to find pattern like "observation_id": "xxx" or just the ID
                        for obs in observations:
                            if obs['observation_id'] in result:
                                self.successful_api_requests += 1
                                return obs['observation_id']
                        
                        # If exact match not found, return the cleaned result
                        # Remove quotes and extra characters
                        cleaned = result.replace('"', '').replace("'", '').strip()
                        # Verify it's a valid observation_id
                        valid_ids = [obs['observation_id'] for obs in observations]
                        if cleaned in valid_ids:
                            self.successful_api_requests += 1
                            return cleaned
                        
                        self.failed_api_requests += 1
                        return None
                    else:
                        self.failed_api_requests += 1
                        print(f"API error: {response.status}")
                        return None
                        
            except Exception as e:
                self.failed_api_requests += 1
                print(f"Error in API call attempt {attempt}: {e}")
                return None
    
    async def select_observation_for_image(self, session: aiohttp.ClientSession, 
                                         image_path: str, observations: List[Dict]) -> Optional[Dict]:
        """Make 5 API calls and select most frequent observation"""
        self.images_processed += 1
        results = []
        
        tasks = []
        for i in range(5):
            task = self.query_together_ai(session, image_path, observations, i + 1)
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks)
        
        # Filter out None responses
        valid_responses = [r for r in responses if r is not None]
        
        if not valid_responses:
            print(f"No valid responses for image {image_path}")
            return None
        
        # Count frequencies
        counter = Counter(valid_responses)
        most_common = counter.most_common()
        
        if most_common:
            most_common_id, count = most_common[0]
            
            # Check if we have consensus (appears at least 2 times)
            if count >= 2:
                # Find the observation with this ID
                for obs in observations:
                    if obs['observation_id'] == most_common_id:
                        self.images_with_consensus += 1
                        print(f"✓ Consensus achieved for {image_path}: {most_common_id} ({count}/5 votes)")
                        return obs
            else:
                # No consensus - randomly select from valid responses
                selected_id = random.choice(valid_responses)
                for obs in observations:
                    if obs['observation_id'] == selected_id:
                        self.images_with_random_selection += 1
                        print(f"⚡ Random selection for {image_path}: {selected_id} (no consensus: {counter})")
                        return obs
        
        print(f"✗ No valid responses for image {image_path}")
        return None
    
    async def _process_species_images(self, session: aiohttp.ClientSession, species: str, 
                                     image_paths: List[str], observations: List[Dict],
                                     process_callback=None):
        """Common logic for processing images of a species
        
        Args:
            session: aiohttp session
            species: Species name
            image_paths: List of image paths for the species
            observations: List of observations for the species
            process_callback: Callback function to process each (image_path, observation) pair
        """
        self.species_processed += 1
        
        # Limit images per species
        selected_images = image_paths[:self.photos_per_species]
        
        # Filter out already processed images for CSV mode
        if hasattr(self, 'processed_images'):
            images_to_process = [img for img in selected_images if img not in self.processed_images]
            skipped_count = len(selected_images) - len(images_to_process)
            
            if skipped_count > 0:
                print(f"\n🌿 Processing species {self.species_processed}/{self.total_species}: {species} ({len(images_to_process)} new, {skipped_count} skipped)")
            else:
                print(f"\n🌿 Processing species {self.species_processed}/{self.total_species}: {species} ({len(images_to_process)} images)")
        else:
            images_to_process = selected_images
            print(f"\n🌿 Processing species {self.species_processed}/{self.total_species}: {species} ({len(images_to_process)} images)")
        
        if not images_to_process:
            print(f"  ✓ All images already processed for {species}")
            return []
        
        results = []
        for idx, image_path in enumerate(images_to_process, 1):
            print(f"  📸 Image {idx}/{len(images_to_process)}: {os.path.basename(image_path)}")
            observation = await self.select_observation_for_image(session, image_path, observations)
            
            if observation:
                if process_callback:
                    result = process_callback(image_path, observation)
                    if result:
                        results.append(result)
                else:
                    results.append((image_path, observation))
        
        return results
    
    async def process_species_batch_csv(self, session: aiohttp.ClientSession, 
                                      species_batch: List[Tuple[str, List[str]]]):
        """Process a batch of species and save to CSV"""
        def save_callback(image_path, observation):
            # Save to CSV immediately
            self.save_observation_to_csv(image_path, observation)
            # Add to processed set
            self.processed_images.add(image_path)
            return None  # No need to collect results
        
        for species, image_paths in species_batch:
            observations = self.observations_by_species[species]
            await self._process_species_images(session, species, image_paths, observations, save_callback)
    
    async def process_species_batch(self, session: aiohttp.ClientSession, 
                                  species_batch: List[Tuple[str, List[str]]]) -> List[Dict]:
        """Process a batch of species concurrently"""
        dataset_rows = []
        
        def create_row_callback(image_path, observation):
            # Load and preprocess image
            try:
                image = self.preprocess_image(image_path)
                
                # Create dataset row
                row = self.create_dataset_row(image, observation)
                return row
                
            except Exception as e:
                print(f"Error loading/preprocessing image {image_path}: {e}")
                return None
        
        for species, image_paths in species_batch:
            observations = self.observations_by_species[species]
            rows = await self._process_species_images(session, species, image_paths, observations, create_row_callback)
            dataset_rows.extend(rows)
        
        return dataset_rows
    
    async def build_dataset(self) -> Dataset:
        """Build the complete dataset"""
        all_rows = []
        
        # Initialize semaphore for rate limiting
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Create batches for concurrent processing
        species_items = list(self.species_to_images.items())
        self.total_species = len(species_items)
        batch_size = 10  # Process 10 species concurrently
        
        # Each image makes 5 API calls, so total concurrent connections = max_concurrent_requests * 5
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_requests * 5)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Process in batches
            for i in tqdm(range(0, len(species_items), batch_size), desc="Processing species batches"):
                batch = species_items[i:i + batch_size]
                batch_rows = await self.process_species_batch(session, batch)
                all_rows.extend(batch_rows)
                
                # Print progress statistics
                self.print_progress_summary(i//batch_size + 1, (len(species_items) + batch_size - 1)//batch_size, len(all_rows))
        
        # Create HuggingFace dataset
        if not all_rows:
            raise ValueError("No valid rows created for dataset")
        
        print(f"Creating dataset with {len(all_rows)} rows")
        
        # Define features
        features = self.get_dataset_features()
        
        dataset = Dataset.from_list(all_rows, features=features)
        return dataset
    
    async def process_and_save_to_csv(self):
        """Process all images and save results to CSV (Stage 1)"""
        # Load existing results first
        self.load_processed_images()
        
        # Initialize semaphore for rate limiting
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Create batches for concurrent processing
        species_items = list(self.species_to_images.items())
        self.total_species = len(species_items)
        batch_size = 10  # Process 10 species concurrently
        
        # Reset progress tracking for this run
        self.species_processed = 0
        self.images_processed = 0
        self.images_with_consensus = 0
        self.images_with_random_selection = 0
        self.total_api_requests = 0
        self.successful_api_requests = 0
        self.failed_api_requests = 0
        
        # Count total images to process
        total_images_available = sum(len(images[:self.photos_per_species]) for _, images in species_items)
        already_processed = len(self.processed_images)
        to_process = total_images_available - already_processed
        
        print(f"\n📊 Resume Status:")
        print(f"   Total images available: {total_images_available}")
        print(f"   Already processed: {already_processed}")
        print(f"   To process: {to_process}")
        print("="*60)
        
        # Each image makes 5 API calls, so total concurrent connections = max_concurrent_requests * 5
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_requests * 5)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Process in batches
            for i in tqdm(range(0, len(species_items), batch_size), desc="Processing species batches"):
                batch = species_items[i:i + batch_size]
                await self.process_species_batch_csv(session, batch)
                
                # Print progress statistics
                self.print_progress_summary(i//batch_size + 1, (len(species_items) + batch_size - 1)//batch_size)
                elapsed = time.time() - self.start_time
                print(f"   Images processed: {self.images_processed} (skipped {len(self.processed_images) - self.images_processed} existing)")
                if self.images_processed > 0 and elapsed > 0:
                    print(f"   Estimated remaining: {(to_process - self.images_processed) / (self.images_processed / elapsed) / 60:.1f} min")
        
        print(f"\n✅ Stage 1 complete! Results saved to {self.csv_output_file}")
    
    def create_dataset_from_csv(self, csv_file: str, batch_size: int = 1000) -> Dataset:
        """Create HuggingFace dataset from CSV file (Stage 2) with batched processing"""
        print(f"Loading observations from {csv_file}...")
        observations = self.load_observations_from_csv(csv_file)
        
        if not observations:
            raise ValueError(f"No valid observations found in {csv_file}")
        
        print(f"Loaded {len(observations)} observations")
        print(f"Creating dataset in batches of {batch_size}...")
        
        # Define features
        features = self.get_dataset_features()
        
        # Process in batches to avoid memory issues
        dataset_parts = []
        total_errors = 0
        successful_rows = 0
        
        for i in tqdm(range(0, len(observations), batch_size), desc="Processing batches"):
            batch = observations[i:i + batch_size]
            batch_rows = []
            batch_errors = 0
            
            for image_path, observation in batch:
                try:
                    # Validate observation data
                    if not observation.get("observation_id") or not observation.get("species"):
                        batch_errors += 1
                        print(f"⚠️  Skipping row with missing required fields: {image_path}")
                        continue
                    
                    # Load and preprocess image
                    image = self.preprocess_image(image_path)
                    
                    # Verify image is valid
                    if image is None or not hasattr(image, 'size'):
                        batch_errors += 1
                        print(f"⚠️  Skipping row with invalid image: {image_path}")
                        continue
                    
                    # Create dataset row with validation
                    row = self.create_dataset_row(image, observation)
                    
                    batch_rows.append(row)
                    successful_rows += 1
                    
                except Exception as e:
                    batch_errors += 1
                    print(f"⚠️  Error processing image {image_path}: {e}")
                    continue
            
            total_errors += batch_errors
            
            if batch_rows:
                # Create dataset from batch and add to list
                try:
                    batch_dataset = Dataset.from_list(batch_rows, features=features)
                    dataset_parts.append(batch_dataset)
                    
                    # Clear batch_rows to free memory
                    del batch_rows
                    
                    print(f"  ✓ Processed batch {i//batch_size + 1}/{(len(observations) + batch_size - 1)//batch_size} - {len(batch_dataset)} rows ({batch_errors} errors)")
                except Exception as e:
                    print(f"  ❌ Failed to create dataset from batch: {e}")
                    total_errors += len(batch)
        
        if not dataset_parts:
            raise ValueError("No valid rows created for dataset")
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total observations: {len(observations)}")
        print(f"   Successful rows: {successful_rows}")
        print(f"   Failed rows: {total_errors}")
        print(f"   Success rate: {successful_rows/len(observations)*100:.1f}%")
        
        print(f"\nConcatenating {len(dataset_parts)} dataset parts...")
        
        # Concatenate all parts into final dataset
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(dataset_parts)
        
        print(f"✅ Created dataset with {len(dataset)} rows")
        return dataset
    
    def validate_dataset(self, dataset: Dataset) -> Tuple[bool, List[str]]:
        """Validate dataset before upload to HuggingFace
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        print("\n🔍 Validating dataset before upload...")
        
        # Check if dataset is empty
        if len(dataset) == 0:
            issues.append("Dataset is empty")
            return False, issues
        
        # Check feature consistency
        expected_features_def = self.get_dataset_features()
        expected_features = {
            "image": DatasetImage,
            "observation_id": Value,
            "species": Value,
            "family": Value,
            "genus": Value,
            "color": Value,
            "inflorescencetype": Value,
            "inflorescence_description": Value,
            "flower_arrangement": Value,
            "flower_density": Value,
            "unique_visual_description": Value,
            "morphological_traits_observable_in_photograph": Value,
            "visual_contrast_with_similar_species": Value
        }
        
        # Verify all expected features are present
        for feature_name, expected_type in expected_features.items():
            if feature_name not in dataset.features:
                issues.append(f"Missing required feature: {feature_name}")
        
        # Sample validation (check first 100 rows or all if less)
        sample_size = min(100, len(dataset))
        print(f"  Checking {sample_size} sample rows...")
        
        invalid_rows = []
        for idx in range(sample_size):
            try:
                row = dataset[idx]
                
                # Check image
                if row.get("image") is None:
                    invalid_rows.append(f"Row {idx}: Missing image")
                    continue
                
                # Try to access the image to ensure it's valid
                try:
                    if hasattr(row["image"], "size"):
                        width, height = row["image"].size
                        if width != 384 or height != 384:
                            invalid_rows.append(f"Row {idx}: Image size is {width}x{height}, expected 384x384")
                except Exception as e:
                    invalid_rows.append(f"Row {idx}: Cannot access image - {str(e)}")
                
                # Check text fields
                for field in ["observation_id", "species", "family", "genus"]:
                    if field in row:
                        if row[field] is None or row[field] == "":
                            invalid_rows.append(f"Row {idx}: Empty required field '{field}'")
                
            except Exception as e:
                invalid_rows.append(f"Row {idx}: Error accessing row - {str(e)}")
        
        if invalid_rows:
            issues.extend(invalid_rows[:10])  # Show first 10 issues
            if len(invalid_rows) > 10:
                issues.append(f"... and {len(invalid_rows) - 10} more issues")
        
        # Try to convert to pandas to check for structural issues
        print("  Testing dataset structure...")
        try:
            # Don't actually convert the whole dataset, just test a small sample
            test_sample = dataset.select(range(min(10, len(dataset))))
            test_df = test_sample.to_pandas()
            del test_df  # Free memory
            print("  ✓ Dataset structure is valid")
        except Exception as e:
            issues.append(f"Cannot convert dataset to pandas: {str(e)}")
        
        # Check dataset info
        try:
            dataset_info = dataset.info
            print(f"  ✓ Dataset info accessible")
        except Exception as e:
            issues.append(f"Cannot access dataset info: {str(e)}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            print("✅ Dataset validation passed!")
        else:
            print(f"❌ Dataset validation failed with {len(issues)} issues")
        
        return is_valid, issues
    
    def fix_dataset_issues(self, dataset: Dataset) -> Dataset:
        """Attempt to fix common dataset issues
        
        Returns:
            Dataset: Fixed dataset
        """
        print("  Scanning dataset for fixable issues...")
        
        # Filter out rows with invalid data
        valid_indices = []
        removed_count = 0
        
        for idx in tqdm(range(len(dataset)), desc="Checking rows"):
            try:
                row = dataset[idx]
                
                # Check if image exists and is accessible
                if row.get("image") is None:
                    removed_count += 1
                    continue
                
                # Try to access image
                try:
                    if hasattr(row["image"], "size"):
                        # Image is valid
                        pass
                    else:
                        removed_count += 1
                        continue
                except:
                    removed_count += 1
                    continue
                
                # Check required text fields
                required_fields = ["observation_id", "species", "family", "genus"]
                skip_row = False
                for field in required_fields:
                    if field not in row or row[field] is None or row[field] == "":
                        skip_row = True
                        break
                
                if skip_row:
                    removed_count += 1
                    continue
                
                # Row is valid
                valid_indices.append(idx)
                
            except Exception as e:
                removed_count += 1
                continue
        
        if removed_count > 0:
            print(f"  ⚠️  Removing {removed_count} invalid rows...")
            dataset = dataset.select(valid_indices)
            print(f"  ✓ Dataset now has {len(dataset)} valid rows")
        else:
            print("  ✓ No invalid rows found")
        
        return dataset
    
    def upload_to_huggingface(self, dataset: Dataset, dataset_name: str, max_shard_size: str = "500MB"):
        """Upload dataset to HuggingFace Hub with smaller shards and progress tracking"""
        import time
        from datetime import datetime
        import sys
        
        # Validate dataset before upload
        is_valid, validation_issues = self.validate_dataset(dataset)
        
        if not is_valid:
            print("\n⚠️  Dataset validation failed! Issues found:")
            for issue in validation_issues:
                print(f"   - {issue}")
            
            # Try to fix common issues
            print("\n🔧 Attempting to fix dataset issues...")
            dataset = self.fix_dataset_issues(dataset)
            
            # Re-validate after fixes
            is_valid, validation_issues = self.validate_dataset(dataset)
            if not is_valid:
                print("\n❌ Dataset still has issues after attempted fixes:")
                for issue in validation_issues:
                    print(f"   - {issue}")
                raise ValueError("Dataset validation failed. Please fix the issues before uploading.")
        
        print(f"\n📤 Preparing upload to HuggingFace Hub...")
        print(f"Dataset name: {dataset_name}")
        print(f"Max shard size: {max_shard_size}")
        print(f"Total rows: {len(dataset):,}")
        
        # Estimate dataset size
        try:
            # Get approximate size by sampling
            sample_size = min(100, len(dataset))
            sample_dataset = dataset.select(range(sample_size))
            sample_bytes = sys.getsizeof(sample_dataset.to_pandas().to_dict('records'))
            estimated_total_bytes = (sample_bytes / sample_size) * len(dataset)
            estimated_gb = estimated_total_bytes / (1024**3)
            print(f"Estimated dataset size: ~{estimated_gb:.2f} GB")
        except:
            print("Could not estimate dataset size")
        
        dataset_dict = DatasetDict({"train": dataset})
        
        # Start timing
        start_time = time.time()
        start_datetime = datetime.now()
        last_update_time = start_time
        
        print(f"\n🚀 Upload started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Custom progress tracking
        class ProgressTracker:
            def __init__(self):
                self.uploaded_shards = 0
                self.start_time = time.time()
                self.last_update = time.time()
                
            def update(self, n=1):
                self.uploaded_shards += n
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Update every 5 seconds
                if current_time - self.last_update > 5:
                    speed = self.uploaded_shards / elapsed if elapsed > 0 else 0
                    eta = (estimated_shards - self.uploaded_shards) / speed if speed > 0 else 0
                    
                    print(f"\r📊 Progress: Shard {self.uploaded_shards}/{estimated_shards} "
                          f"({self.uploaded_shards/estimated_shards*100:.1f}%) | "
                          f"Speed: {speed:.2f} shards/min | "
                          f"ETA: {eta/60:.1f} min", end='', flush=True)
                    
                    self.last_update = current_time
        
        # Estimate number of shards
        shard_size_mb = int(max_shard_size.replace('MB', '').replace('GB', '000'))
        estimated_shards = max(1, int(estimated_gb * 1024 / shard_size_mb)) if 'estimated_gb' in locals() else 10
        
        print(f"Estimated number of shards: ~{estimated_shards}")
        print(f"Starting upload with {max_shard_size} shards...\n")
        
        try:
            # Additional validation before push
            print("\n🔄 Performing final pre-upload checks...")
            
            # Test dataset access
            try:
                test_row = dataset[0]
                print("  ✓ Can access dataset rows")
            except Exception as e:
                print(f"  ❌ Cannot access dataset rows: {e}")
                raise
            
            # Verify dataset can be saved locally (small test)
            try:
                test_dataset = dataset.select(range(min(10, len(dataset))))
                test_dataset.save_to_disk("/tmp/hf_test_dataset", num_proc=1)
                import shutil
                shutil.rmtree("/tmp/hf_test_dataset")
                print("  ✓ Dataset can be serialized")
            except Exception as e:
                print(f"  ❌ Dataset serialization test failed: {e}")
                raise
            
            print("\n🚀 Starting upload...")
            
            # Push with smaller shard size for better progress tracking
            dataset_dict.push_to_hub(
                dataset_name,
                token=self.hf_token,
                private=False,
                max_shard_size=max_shard_size,  # Smaller shards for more granular progress
                commit_message="Upload PlantNet-300K botanical observations dataset",
            )
            
            # Calculate upload statistics
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n\n{'='*60}")
            print(f"✅ Dataset uploaded successfully!")
            print(f"{'='*60}")
            print(f"📍 Dataset URL: https://huggingface.co/datasets/{dataset_name}")
            print(f"⏱️  Upload completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Total upload time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
            print(f"📊 Average speed: {len(dataset)/total_time:.1f} rows/second")
            if 'estimated_gb' in locals():
                print(f"📈 Upload rate: ~{estimated_gb/(total_time/3600):.2f} GB/hour")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n\n❌ Upload failed after {(time.time() - start_time)/60:.2f} minutes")
            print(f"Error: {str(e)}")
            print(f"{'='*60}")
            
            # Specific error handling
            error_msg = str(e).lower()
            if "size of the dataset is not coherent" in error_msg:
                print("\n🔍 Debugging 'dataset size not coherent' error:")
                print("This error typically occurs when:")
                print("1. Dataset has inconsistent row structure")
                print("2. Some images failed to load properly")
                print("3. Dataset metadata is corrupted")
                print("\nSuggested fixes:")
                print("1. Re-run with --stage 2 to recreate dataset from CSV")
                print("2. Use a smaller batch size (--batch-size 500)")
                print("3. Check the CSV file for corrupted entries")
                
                # Try to provide more diagnostic info
                try:
                    print(f"\nDataset diagnostics:")
                    print(f"  - Number of rows: {len(dataset)}")
                    print(f"  - Features: {list(dataset.features.keys())}")
                    print(f"  - First row keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'N/A'}")
                except:
                    print("  - Could not access dataset diagnostics")
            
            raise
    
    async def run(self, stage: str = "both", csv_file: str = "observations.csv", dataset_name: str = None, 
                  batch_size: int = 1000, max_shard_size: str = "500MB"):
        """Main execution flow with 2-stage support"""
        
        # Stage 1: Process and save to CSV
        if stage in ["1", "both"]:
            print("\n🚀 Starting Stage 1: Process images and save to CSV")
            print("="*60)
            
            # Load data
            self.load_observations()
            self.load_plant_files()
            
            # Process and save to CSV
            await self.process_and_save_to_csv()
            
            # Print final statistics for Stage 1
            total_time = time.time() - self.start_time
            print("\n" + "="*60)
            print("🎉 STAGE 1 STATISTICS")
            print("="*60)
            print(f"Total species available: {self.total_species}")
            print(f"Species successfully processed: {self.species_processed}/{self.total_species} ({self.species_processed/self.total_species*100:.1f}%)")
            print(f"Total images processed: {self.images_processed}")
            if self.species_processed > 0:
                print(f"Average images per species: {self.images_processed/self.species_processed:.1f}")
            print(f"Images with consensus: {self.images_with_consensus} ({self.images_with_consensus/self.images_processed*100:.1f}%)")
            print(f"Images with random selection: {self.images_with_random_selection} ({self.images_with_random_selection/self.images_processed*100:.1f}%)")
            print(f"Images failed (no valid responses): {self.images_processed - self.images_with_consensus - self.images_with_random_selection}")
            print(f"Total API requests: {self.total_api_requests}")
            print(f"Successful API requests: {self.successful_api_requests} ({self.successful_api_requests/self.total_api_requests*100:.1f}%)")
            print(f"Failed API requests: {self.failed_api_requests}")
            if self.images_processed > 0:
                print(f"Average time per image: {total_time/self.images_processed:.2f}s")
            if self.species_processed > 0:
                print(f"Average time per species: {total_time/self.species_processed:.2f}s")
            print(f"Total processing time: {total_time/60:.1f} minutes")
            print("="*60)
        
        # Stage 2: Create dataset from CSV
        if stage in ["2", "both"]:
            if not dataset_name:
                raise ValueError("--dataset-name is required when running stage 2")
                
            print("\n🚀 Starting Stage 2: Create dataset from CSV")
            print("="*60)
            
            # Create dataset from CSV
            dataset = self.create_dataset_from_csv(csv_file, batch_size=batch_size)
            
            # Upload to HuggingFace
            self.upload_to_huggingface(dataset, dataset_name, max_shard_size=max_shard_size)
            print("\n✅ Stage 2 complete! Dataset uploaded to HuggingFace Hub.")
        
        print("\n🎉 All stages completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Prepare PlantNet-300K observation dataset")
    parser.add_argument(
        "--photos-per-species", 
        type=int, 
        default=10,
        help="Maximum number of photos to process per species (default: 10)"
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=5,
        help="Maximum number of images to process concurrently. Each image makes 5 API calls, so total concurrent API connections = this value * 5 (default: 5)"
    )
    parser.add_argument(
        "--stage",
        choices=["both", "1", "2"],
        default="both",
        help="Which stage to run: 1 (process and save to CSV), 2 (create dataset from CSV), or both (default: both)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="observations.csv",
        help="CSV file to save/load observations (default: observations.csv)"
    )
    parser.add_argument(
        "--observations-file",
        type=str,
        default="json_observation_list.jsonl",
        help="Path to the JSONL file containing observations (default: json_observation_list.jsonl)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="HuggingFace dataset name (e.g., 'username/dataset-name') - required for stage 2"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing images in stage 2 (default: 1000)"
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="500MB",
        help="Maximum size of each parquet shard for upload (default: 500MB, e.g., 250MB, 1GB)"
    )
    
    args = parser.parse_args()
    
    # Warn if concurrent requests is set too high
    if args.max_concurrent_requests > 10:
        print(f"⚠️  Warning: --max-concurrent-requests is set to {args.max_concurrent_requests}")
        print(f"   This will create up to {args.max_concurrent_requests * 5} concurrent API connections.")
        print("   Consider lowering this value if you experience rate limiting or connection issues.")
        print()
    
    builder = PlantObservationDatasetBuilder(
        photos_per_species=args.photos_per_species,
        max_concurrent_requests=args.max_concurrent_requests,
        observations_file=args.observations_file
    )
    builder.csv_output_file = args.csv_file
    asyncio.run(builder.run(stage=args.stage, csv_file=args.csv_file, dataset_name=args.dataset_name, 
                           batch_size=args.batch_size, max_shard_size=args.max_shard_size))


if __name__ == "__main__":
    main()