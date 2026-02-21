"""
Configuration module for Qwen3.5 Virtual Try-On Pipeline
Loads settings from environment variables and .env file
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class that loads settings from environment variables."""
    
    def __init__(self):
        self._setup_logging()
        self._create_directories()
    
    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    
    @property
    def qwen_model_id(self) -> str:
        """Qwen3.5 multimodal model identifier."""
        return os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-397B-A17B")
    
    @property
    def device(self) -> str:
        """Device to use for computation."""
        device = os.getenv("DEVICE", "auto")
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    # =============================================================================
    # IMAGE PROCESSING SETTINGS
    # =============================================================================
    
    @property
    def image_resolution(self) -> int:
        """Default image resolution for processing."""
    @property
    def image_resolution(self) -> int:
        """Default image resolution for processing."""
        return int(os.getenv("IMAGE_RESOLUTION", "512"))
    
    @property
    def random_seed(self) -> Optional[int]:
        """Random seed for reproducible results."""
        seed_str = os.getenv("RANDOM_SEED", "")
        return int(seed_str) if seed_str else None
    
    # =============================================================================
    # QWEN GENERATION SETTINGS
    # =============================================================================
    
    @property
    def max_new_tokens(self) -> int:
        """Maximum tokens for Qwen responses."""
        return int(os.getenv("MAX_NEW_TOKENS", "2048"))
    
    @property
    def qwen_temperature(self) -> float:
        """Temperature for Qwen generation."""
        return float(os.getenv("QWEN_TEMPERATURE", "0.7"))

    # =============================================================================
    # FILE PATHS
    # =============================================================================
    
    @property
    def person_images_dir(self) -> Path:
        """Directory for person images."""
        return Path(os.getenv("PERSON_IMAGES_DIR", "./input/person_images"))
    
    @property
    def garment_images_dir(self) -> Path:
        """Directory for garment images."""
        return Path(os.getenv("GARMENT_IMAGES_DIR", "./input/garment_images"))
    

    
    @property
    def output_dir(self) -> Path:
        """Main output directory."""
        return Path(os.getenv("OUTPUT_DIR", "./outputs"))
    
    @property
    def results_dir(self) -> Path:
        """Directory for final results."""
        return Path(os.getenv("RESULTS_DIR", "./results"))
    
    @property
    def analysis_dir(self) -> Path:
        """Directory for analysis results."""
        return Path(os.getenv("ANALYSIS_DIR", "./analysis"))
    
    @property
    def comparison_dir(self) -> Path:
        """Directory for comparison images."""
        return Path(os.getenv("COMPARISON_DIR", "./comparisons"))
    
    @property
    def cache_dir(self) -> Path:
        """Directory for model cache."""
        return Path(os.getenv("CACHE_DIR", "./models_cache"))
    
    # =============================================================================
    # DEFAULT FILE NAMES
    # =============================================================================
    
    @property
    def default_person_image(self) -> str:
        """Default person image filename."""
        return os.getenv("DEFAULT_PERSON_IMAGE", "person1.jpg")
    
    @property
    def default_garment_image(self) -> str:
        """Default garment image filename."""
        return os.getenv("DEFAULT_GARMENT_IMAGE", "shirt1.jpg")
    
    @property
    def default_person_description(self) -> str:
        """Default person description."""
        return os.getenv("DEFAULT_PERSON_DESCRIPTION", "person wearing a white t-shirt")

    @property
    def output_filename_pattern(self) -> str:
        """Pattern for output filenames."""
        return os.getenv("OUTPUT_FILENAME_PATTERN", "{person_name}_wearing_{garment_name}")
    
    # =============================================================================
    # PROCESSING OPTIONS
    # =============================================================================
    
    @property
    def save_analysis(self) -> bool:
        """Save analysis results by default."""
        return self._get_bool("SAVE_ANALYSIS", True)
    
    @property
    def show_summary(self) -> bool:
        """Show analysis summary by default."""
        return self._get_bool("SHOW_SUMMARY", True)
    
    @property
    def create_comparison(self) -> bool:
        """Create comparison grids by default."""
        return self._get_bool("CREATE_COMPARISON", True)
    

    
    # =============================================================================
    # LOGGING AND DEBUG
    # =============================================================================
    
    @property
    def log_level(self) -> str:
        """Log level."""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def detailed_logging(self) -> bool:
        """Enable detailed logging."""
        return self._get_bool("DETAILED_LOGGING", False)
    
    @property
    def save_intermediate_results(self) -> bool:
        """Save intermediate results for debugging."""
        return self._get_bool("SAVE_INTERMEDIATE_RESULTS", False)
    
    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    
    @property
    def enable_memory_optimization(self) -> bool:
        """Enable memory optimization for Qwen."""
        return self._get_bool("ENABLE_MEMORY_OPTIMIZATION", True)
    
    @property
    def num_threads(self) -> int:
        """Number of CPU threads for processing."""
        return int(os.getenv("NUM_THREADS", "4"))
    
    # =============================================================================
    # QUALITY SETTINGS
    # =============================================================================
    
    @property
    def high_quality_mode(self) -> bool:
        """High quality mode."""
        return self._get_bool("HIGH_QUALITY_MODE", False)
    
    @property
    def enable_detailed_prompting(self) -> bool:
        """Enable detailed prompting for better results."""
        return self._get_bool("ENABLE_DETAILED_PROMPTING", True)
    
    # =============================================================================
    # API KEYS
    # =============================================================================
    
    @property
    def huggingface_token(self) -> Optional[str]:
        """Hugging Face token."""
        return os.getenv("HUGGINGFACE_TOKEN")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API key."""
        return os.getenv("OPENAI_API_KEY")
    
    # =============================================================================
    # EXPERIMENTAL FEATURES
    # =============================================================================
    
    @property
    def enable_experimental(self) -> bool:
        """Enable experimental features."""
        return self._get_bool("ENABLE_EXPERIMENTAL", False)
    
    @property
    def use_advanced_prompting(self) -> bool:
        """Enable advanced prompting techniques."""
        return self._get_bool("USE_ADVANCED_PROMPTING", False)
    

    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if self.detailed_logging:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        
        logging.basicConfig(
            level=log_level,
            format=format_str,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("virtual_tryon.log") if self.detailed_logging else logging.NullHandler()
            ]
        )
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.output_dir,
            self.results_dir, 
            self.analysis_dir,
            self.comparison_dir,
            self.cache_dir,
            self.person_images_dir,
            self.garment_images_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_output_filename(self, person_name: str, garment_name: str) -> str:
        """Generate output filename based on pattern."""
        return self.output_filename_pattern.format(
            person_name=person_name,
            garment_name=garment_name
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        # Get all properties
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                try:
                    value = getattr(self, attr_name)
                    # Convert Path objects to strings
                    if isinstance(value, Path):
                        value = str(value)
                    config_dict[attr_name] = value
                except Exception:
                    # Skip properties that can't be accessed
                    pass
        
        return config_dict
    
    def validate_configuration(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check if CUDA is requested but not available
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append("CUDA requested but not available")
            except ImportError:
                issues.append("PyTorch not installed but CUDA device specified")
        
        # Check if required directories exist for input
        if not self.person_images_dir.exists():
            issues.append("Person images directory not found")
        
        # Validate numeric ranges
        if not (0.0 <= self.qwen_temperature <= 2.0):
            issues.append(f"Qwen temperature {self.qwen_temperature} should be between 0.0 and 2.0")
        
        if self.image_resolution < 128 or self.image_resolution > 2048:
            issues.append(f"Image resolution {self.image_resolution} should be between 128 and 2048")
        
        return issues
    
    def print_configuration(self):
        """Print current configuration."""
        print("=" * 60)
        print("QWEN3.5 VIRTUAL TRY-ON CONFIGURATION")
        print("=" * 60)
        
        config_dict = self.to_dict()
        
        categories = {
            "Model Settings": ["qwen_model_id", "device"],
            "Generation Settings": ["max_new_tokens", "qwen_temperature"],
            "Image Processing": ["image_resolution", "random_seed"],
            "Directory Paths": ["output_dir", "results_dir", "person_images_dir", "garment_images_dir"],
            "Processing Options": ["save_analysis", "show_summary", "create_comparison"],
            "Performance": ["enable_memory_optimization", "num_threads"],
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            print("-" * len(category))
            for key in keys:
                if key in config_dict:
                    value = config_dict[key]
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    print(f"  {key}: {value}")
        
        # Show validation issues if any
        issues = self.validate_configuration()
        if issues:
            print(f"\n⚠️  Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ Configuration is valid")

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def reload_config():
    """Reload configuration from environment variables."""
    global config
    load_dotenv(override=True)  # Reload .env file
    config = Config()
    return config