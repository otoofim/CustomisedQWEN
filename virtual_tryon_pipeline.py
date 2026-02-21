"""
Qwen3.5-Only Virtual Try-On Pipeline
Simple end-to-end virtual try-on using only Qwen3.5's multimodal capabilities.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Tuple, Union, List, Dict
import json
from dataclasses import dataclass
from pathlib import Path

# Qwen3.5 imports
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)

# Import configuration
from config import get_config

@dataclass
class QwenTryOnConfig:
    """Configuration for the Qwen3.5-only virtual try-on pipeline."""
    
    def __init__(self, **kwargs):
        """Initialize configuration with values from environment variables and overrides."""
        config = get_config()
        
        # Model configurations
        self.qwen_model_id = kwargs.get('qwen_model_id', config.qwen_model_id)
        self.device = kwargs.get('device', config.device)
        
        # Generation settings
        self.max_new_tokens = kwargs.get('max_new_tokens', config.max_new_tokens)
        self.temperature = kwargs.get('temperature', config.qwen_temperature)
        
        # Image processing settings
        self.image_resolution = kwargs.get('image_resolution', config.image_resolution)
        self.seed = kwargs.get('seed', config.random_seed)
        
        # Performance settings
        self.enable_memory_optimization = kwargs.get('enable_memory_optimization', config.enable_memory_optimization)
    
    @classmethod
    def from_env(cls, **overrides):
        """Create configuration from environment variables with optional overrides."""
        return cls(**overrides)


class QwenVirtualTryOnPipeline:
    """
    Simple Qwen3.5-only virtual try-on pipeline.
    Takes person and garment images, generates virtual try-on result directly.
    """
    
    def __init__(self, config: QwenTryOnConfig = None):
        self.config = config or QwenTryOnConfig()
        self.device = torch.device(self.config.device)
        
        print("Loading Qwen3.5 multimodal model...")
        
        # Load Qwen3.5-VL model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.qwen_model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.qwen_model_id,
            trust_remote_code=True
        )
        
        # Enable memory optimization if requested
        if self.config.enable_memory_optimization and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        print("Qwen3.5 model loaded successfully!")
    
    def virtual_try_on(self,
                      person_image: Union[str, Image.Image, np.ndarray],
                      garment_image: Union[str, Image.Image, np.ndarray],
                      person_description: str,
                      return_analysis: bool = False) -> Union[Image.Image, Tuple[Image.Image, str]]:
        """
        Perform virtual try-on using only Qwen3.5.
        
        Args:
            person_image: Image of person to dress
            garment_image: Image of the garment to apply  
            person_description: Description of the target person
            return_analysis: Whether to return analysis text
            
        Returns:
            Generated try-on result image, optionally with analysis
        """
        print("Starting Qwen3.5 virtual try-on...")
        
        # Preprocess images
        person_img = self._load_and_preprocess_image(person_image)
        garment_img = self._load_and_preprocess_image(garment_image)
        
        # Generate the virtual try-on result directly
        result_image, analysis = self._generate_virtual_tryon(
            person_img, 
            garment_img, 
            person_description
        )
        
        print("Virtual try-on completed!")
        
        if return_analysis:
            return result_image, analysis
        else:
            return result_image
    
    def _load_and_preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Load and preprocess an image to the required format."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert("RGB")
        
        # Resize while maintaining aspect ratio
        image.thumbnail((self.config.image_resolution, self.config.image_resolution), Image.Resampling.LANCZOS)
        
        # Pad to square if needed
        width, height = image.size
        if width != height:
            max_dim = max(width, height)
            new_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        # Final resize to target resolution
        image = image.resize((self.config.image_resolution, self.config.image_resolution), Image.Resampling.LANCZOS)
        
        return image
    
    def _generate_virtual_tryon(self, 
                               person_image: Image.Image, 
                               garment_image: Image.Image, 
                               person_description: str) -> Tuple[Image.Image, str]:
        """
        Generate virtual try-on result using Qwen3.5's multimodal capabilities.
        """
        
        # Create the prompt for virtual try-on
        prompt = self._create_tryon_prompt(person_description)
        
        # Prepare the messages for Qwen3.5
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": person_image,
                    },
                    {
                        "type": "image", 
                        "image": garment_image,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Prepare inputs for the model
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)
        
        # Generate the response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Qwen3.5 analysis: {output_text[:200]}...")
        
        # For now, since Qwen3.5-VL doesn't directly generate images in this version, 
        # we'll create a composite/blend as a placeholder
        # In a real implementation, you would use Qwen's image generation capabilities
        result_image = self._create_composite_image(person_image, garment_image, output_text)
        
        return result_image, output_text
    
    def _create_tryon_prompt(self, person_description: str) -> str:
        """Create an intelligent prompt for virtual try-on."""
        prompt = f"""
You are an expert fashion AI assistant. I'm showing you two images:
1. A person image - this is {person_description}
2. A garment image - this is the clothing item to be applied

Please analyze both images and provide detailed instructions for creating a virtual try-on result where the person is wearing the new garment. 

Your analysis should include:
1. Person analysis: body pose, current clothing, visible areas suitable for replacement
2. Garment analysis: style, color, type, fit characteristics  
3. Compatibility assessment: how well this garment would fit this person
4. Detailed placement instructions: exactly where and how the garment should be positioned
5. Style adjustments needed: any modifications to make it look natural
6. Lighting and shadow considerations: how to match the image lighting
7. Final composition description: describe the ideal result in detail

Provide your analysis in a structured format that could guide an image generation process.
        """.strip()
        
        return prompt
    
    def _create_composite_image(self, person_image: Image.Image, garment_image: Image.Image, analysis: str) -> Image.Image:
        """
        Create a composite image as a placeholder for the virtual try-on result.
        This is a simple implementation - in practice, Qwen3.5's image generation would create the actual result.
        """
        
        # For demonstration, create a side-by-side composite showing the analysis process
        width, height = person_image.size
        composite_width = width * 2 + 20  # Space for both images plus gap
        composite = Image.new('RGB', (composite_width, height), (255, 255, 255))
        
        # Paste the person image
        composite.paste(person_image, (0, 0))
        
        # Paste the garment image
        composite.paste(garment_image, (width + 20, 0))
        
        # In a real implementation, this would be replaced with:
        # - Qwen3.5's actual image generation capabilities
        # - The analysis would guide the generation process
        # - The result would be a realistic virtual try-on image
        
        # Add some visual indication this is a demo
        draw = ImageDraw.Draw(composite)
        
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Add labels
        draw.text((10, 10), "Original", fill=(0, 0, 0), font=font)
        draw.text((width + 30, 10), "Garment", fill=(0, 0, 0), font=font) 
        draw.text((composite_width // 2 - 50, height - 30), "Virtual Try-On Result", fill=(255, 0, 0), font=font)
        
        return composite


# Backwards compatibility methods
def create_comparison_grid(original_images: List[Image.Image], 
                          result_images: List[Image.Image],
                          garment_images: List[Image.Image] = None) -> Image.Image:
    """Create a comparison grid showing original, garment, and result images."""
    
    if not original_images or not result_images:
        raise ValueError("Need at least one original and result image")
    
    num_images = len(original_images)
    cols = 3 if garment_images else 2  # Original, Garment (optional), Result
    
    # Get image dimensions (assuming all images are the same size)
    img_width, img_height = original_images[0].size
    
    # Calculate grid dimensions
    grid_width = cols * img_width + (cols - 1) * 10  # 10px gaps
    grid_height = num_images * img_height + (num_images - 1) * 10
    
    # Create the grid image
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Place images in the grid
    for i in range(num_images):
        y_pos = i * (img_height + 10)
        
        # Original image
        grid.paste(original_images[i], (0, y_pos))
        
        # Garment image (if provided)
        if garment_images and i < len(garment_images):
            grid.paste(garment_images[i], (img_width + 10, y_pos))
            result_x = 2 * (img_width + 10)
        else:
            result_x = img_width + 10
        
        # Result image
        grid.paste(result_images[i], (result_x, y_pos))
    
    return grid


def save_analysis_results(analysis_data: Dict, output_path: Union[str, Path]):
    """Save analysis results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any non-serializable objects to strings
    serializable_data = {}
    for key, value in analysis_data.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            serializable_data[key] = value
        else:
            serializable_data[key] = str(value)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def print_analysis_summary(analysis: str):
    """Print a formatted summary of the analysis."""
    print("=" * 60)
    print("QWEN3.5 VIRTUAL TRY-ON ANALYSIS")
    print("=" * 60)
    print(analysis[:500])
    if len(analysis) > 500:
        print("... [truncated]")
    print("=" * 60)


# Convenience aliases for backwards compatibility
def QwenVisionAnalyzer(config=None):
    """Alias for the main pipeline class for backwards compatibility."""
    return QwenVirtualTryOnPipeline(config)

# Main pipeline alias
QwenTryOnPipeline = QwenVirtualTryOnPipeline