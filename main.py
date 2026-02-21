"""
Main Application for Qwen3.5-Powered Virtual Try-On Pipeline
Intelligent clothing replacement using multimodal AI analysis.

Usage:
    python main.py --person_image path/to/person.jpg --garment_image path/to/garment.jpg --person_description "person wearing a blue shirt"
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import json

from virtual_tryon_pipeline import (
    QwenVirtualTryOnPipeline,
    QwenTryOnConfig,
    create_comparison_grid,
    save_analysis_results,
    print_analysis_summary
)
from config import get_config

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-Only Virtual Try-On Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Copy .env.example to .env and customize values for default settings.
  Command line arguments will override .env file settings.

Example usage:
  # Using .env file defaults (copy .env.example to .env and set defaults)
  python main.py
  
  # Override person description only
  python main.py --person_description "person wearing a blue dress"
  
  # Override specific images
  python main.py --person_image person.jpg --garment_image shirt.jpg
  
  # Full customization
  python main.py --person_image person.jpg --garment_image shirt.jpg --person_description "person wearing a white t-shirt" --temperature 0.5
        """
    )
    
    # Get default values from environment
    env_config = get_config()
    
    # Image arguments (optional - will use .env defaults if not provided)
    parser.add_argument(
        "--person_image",
        help=f"Path to the person image file (default: {env_config.person_images_dir / env_config.default_person_image})"
    )
    parser.add_argument(
        "--garment_image",
        help=f"Path to the garment image file (default: {env_config.garment_images_dir / env_config.default_garment_image})"
    )
    parser.add_argument(
        "--person_description",
        help=f"Description of the target person (default: '{env_config.default_person_description}')"
    )
    
    # Optional arguments with defaults from environment
    parser.add_argument(
        "--output_dir", 
        default=str(env_config.output_dir),
        help=f"Directory to save output images and analysis (default: {env_config.output_dir})"
    )
    parser.add_argument(
        "--qwen_model", 
        default=env_config.qwen_model_id,
        help=f"Qwen3.5 model to use for analysis (default: {env_config.qwen_model_id})"
    )
    parser.add_argument(
        "--image_resolution", 
        type=int, 
        default=env_config.image_resolution,
        help=f"Target image resolution (default: {env_config.image_resolution})"
    )
    parser.add_argument(
        "--use_pose_control", 
        action="store_true",
        help="DEPRECATED: Pose control is no longer used in simplified pipeline"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=env_config.random_seed,
        help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--device", 
        choices=["cuda", "cpu", "auto"],
        default=env_config.device,
        help=f"Device to use for computation (default: {env_config.device})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=env_config.qwen_temperature,
        help=f"Temperature for Qwen generation (default: {env_config.qwen_temperature})"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=env_config.max_new_tokens,
        help=f"Maximum tokens for Qwen responses (default: {env_config.max_new_tokens})"
    )
    parser.add_argument(
        "--save_analysis", 
        action="store_true",
        default=env_config.save_analysis,
        help=f"Save detailed analysis results to JSON file (default: {env_config.save_analysis})"
    )
    
    parser.add_argument(
        "--show_summary", 
        action="store_true",
        default=env_config.show_summary,
        help=f"Show analysis summary (default: {env_config.show_summary})"
    )
    
    args = parser.parse_args()
    
    # Use .env defaults for image paths if not provided
    person_image = args.person_image
    if not person_image:
        person_image = str(env_config.person_images_dir / env_config.default_person_image)
        print(f"Using default person image: {person_image}")
    
    garment_image = args.garment_image 
    if not garment_image:
        garment_image = str(env_config.garment_images_dir / env_config.default_garment_image)
        print(f"Using default garment image: {garment_image}")
    
    person_description = args.person_description
    if not person_description:
        person_description = env_config.default_person_description
        print(f"Using default person description: '{person_description}'")
    
    # Validate input files
    if not os.path.exists(person_image):
        print(f"Error: Person image file '{person_image}' not found.")
        sys.exit(1)
        
    if not os.path.exists(garment_image):
        print(f"Error: Garment image file '{garment_image}' not found.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure device
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Create configuration
    config = QwenTryOnConfig(
        qwen_model_id=args.qwen_model,
        device=device,
        image_resolution=args.image_resolution,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        seed=args.seed
    )
    
    try:
        # Initialize pipeline
        print("Initializing Qwen3.5 Virtual Try-On Pipeline...")
        pipeline = QwenVirtualTryOnPipeline(config)
        
        # Perform virtual try-on with analysis
        print(f"\nProcessing images:")
        print(f"  Person: {person_image}")
        print(f"  Garment: {garment_image}")
        print(f"  Description: {person_description}")
        
        result_image, analysis = pipeline.virtual_try_on(
            person_image=person_image,
            garment_image=garment_image, 
            person_description=person_description,
            return_analysis=True
        )
        
        # Save results
        person_img = Image.open(person_image).convert("RGB")
        garment_img = Image.open(garment_image).convert("RGB")
        
        # Create output filenames
        person_name = Path(person_image).stem
        garment_name = Path(garment_image).stem
        output_prefix = f"{person_name}_wearing_{garment_name}"
        
        # Save the main result
        result_path = os.path.join(args.output_dir, f"{output_prefix}_result.png")
        result_image.save(result_path)
        print(f"\nResult saved: {result_path}")
        
        # Create and save comparison grid
        comparison_grid = create_comparison_grid(
            [person_img], 
            [result_image],
            [garment_img]
        )
        comparison_path = os.path.join(args.output_dir, f"{output_prefix}_comparison.png")
        comparison_grid.save(comparison_path)
        print(f"Comparison grid saved: {comparison_path}")
        
        # Save analysis results if requested
        if args.save_analysis:
            analysis_path = os.path.join(args.output_dir, f"{output_prefix}_analysis.json")
            analysis_data = {"analysis": analysis, "person_description": person_description}
            save_analysis_results(analysis_data, analysis_path)
        
        # Show analysis summary if requested
        if args.show_summary:
            print_analysis_summary(analysis)
        
        print(f"\n✅ Virtual try-on completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during virtual try-on: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_demo():
    """Run a demo with example images (if available)."""
    print("Qwen3.5 Virtual Try-On Demo")
    print("=" * 40)
    
    # Check for demo images
    demo_dir = "demo_images"
    if not os.path.exists(demo_dir):
        print(f"Demo directory '{demo_dir}' not found.")
        print("Please run with your own images using: python main.py --person_image <path> --garment_image <path> --person_description '<description>'")
        return
    
    # Look for demo images
    person_images = list(Path(demo_dir).glob("person_*.jpg")) + list(Path(demo_dir).glob("person_*.png"))
    garment_images = list(Path(demo_dir).glob("garment_*.jpg")) + list(Path(demo_dir).glob("garment_*.png"))
    
    if not person_images or not garment_images:
        print("Demo images not found. Please add:")
        print("  - demo_images/person_*.jpg (person images)")
        print("  - demo_images/garment_*.jpg (garment images)")
        return
    
    # Run demo with first available images
    person_image = person_images[0]
    garment_image = garment_images[0]
    
    print(f"Using demo images:")
    print(f"  Person: {person_image}")
    print(f"  Garment: {garment_image}")
    
    config = QwenTryOnConfig()
    pipeline = QwenVirtualTryOnPipeline(config)
    
    result_image, analysis_data = pipeline.intelligent_try_on(
        person_image=str(person_image),
        garment_image=str(garment_image),
        person_description="person wearing casual clothes",
        return_analysis=True
    )
    
    # Save demo results
    os.makedirs("demo_outputs", exist_ok=True)
    result_image.save("demo_outputs/demo_result.png")
    
    print_analysis_summary(analysis_data)
    print("\n✅ Demo completed! Check demo_outputs/demo_result.png")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("Qwen3.5-Powered Virtual Try-On Pipeline")
        print("=" * 50)
        print("\nUsage:")
        print("  python main.py --person_image <path> --garment_image <path> --person_description '<description>'")
        print("\nExample:")
        print("  python main.py --person_image person.jpg --garment_image shirt.jpg --person_description 'person wearing a blue shirt'")
        print("\nFor demo (if demo images available):")
        print("  python main.py --demo")
        print("\nFor full help:")
        print("  python main.py --help")
        
    elif len(sys.argv) == 2 and sys.argv[1] == "--demo":
        run_demo()
    else:
        main()
