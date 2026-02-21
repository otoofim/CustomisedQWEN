# Qwen3.5-Only Virtual Try-On Pipeline

A simple, streamlined virtual try-on system using only Qwen3.5's multimodal capabilities. This pipeline leverages Qwen3.5-VL's vision-language understanding to create virtual try-on results directly, without the complexity of multiple models.

## 🌟 Features

- **End-to-End Simplicity**: Single Qwen3.5 model handles the entire pipeline
- **Intelligent Analysis**: Deep understanding of person and garment compatibility
- **Direct Generation**: No intermediate models or complex segmentation
- **Fast Processing**: Streamlined workflow with minimal dependencies
- **Easy Setup**: Simple installation and configuration
- **Detailed Feedback**: Comprehensive analysis and recommendations

## 🏗️ Architecture

```
Person Image + Garment Image + Description
                    ↓
        Qwen3.5 Multimodal Model
         ├── Vision Understanding
         ├── Compatibility Analysis  
         └── Virtual Try-On Generation
                    ↓
           Final Try-On Result
```

**Simple and Clean**: Just one powerful model doing all the work!

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd qwen-virtual-tryon
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

### Basic Usage

```python
from virtual_tryon_pipeline import QwenVirtualTryOnPipeline

# Initialize pipeline
pipeline = QwenVirtualTryOnPipeline()

# Perform virtual try-on
result_image, analysis = pipeline.virtual_try_on(
    person_image="path/to/person.jpg",
    garment_image="path/to/garment.jpg", 
    person_description="person wearing a blue shirt",
    return_analysis=True
)

# Save result
result_image.save("tryon_result.png")
```

### Environment Configuration

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env with your preferences:**
   ```bash
   # Model settings
   QWEN_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
   DEVICE=auto
   
   # Default paths and settings
   PERSON_IMAGES_DIR=./input/person_images
   GARMENT_IMAGES_DIR=./input/garment_images
   DEFAULT_PERSON_IMAGE=person1.jpg
   DEFAULT_GARMENT_IMAGE=shirt1.jpg
   DEFAULT_PERSON_DESCRIPTION=person wearing a white t-shirt
   
   # Generation settings
   IMAGE_RESOLUTION=512
   QWEN_TEMPERATURE=0.7
   MAX_NEW_TOKENS=2048
   ```

### Command Line Usage

**Simplest usage (uses .env defaults):**
```bash
# Uses all defaults from .env file
python main.py
```

**Override specific settings:**
```bash
# Custom description only
python main.py --person_description "person wearing a blue dress"

# Custom images
python main.py --person_image custom_person.jpg --garment_image custom_shirt.jpg

# Full customization
python main.py \
    --person_image person.jpg \
    --garment_image dress.jpg \
    --person_description "woman in casual attire" \
    --output_dir results \
    --image_resolution 768 \
    --temperature 0.5 \
    --save_analysis \
    --show_summary
```

**Available options:**
```bash
python main.py --help
```

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: Works on CPU but significantly slower
- **RAM**: 16GB+ recommended

### Software Dependencies
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- OpenCV, Pillow, NumPy
- python-dotenv

## 🔧 Configuration

The pipeline can be customized through `QwenTryOnConfig`:

```python
from virtual_tryon_pipeline import QwenTryOnConfig

config = QwenTryOnConfig(
    qwen_model_id="Qwen/Qwen2-VL-7B-Instruct",  # Qwen model
    image_resolution=512,                        # Output resolution  
    temperature=0.7,                            # Analysis temperature
    max_new_tokens=2048,                        # Response length
    seed=42                                     # Reproducibility
)
```

### Key Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `qwen_model_id` | `Qwen/Qwen2-VL-7B-Instruct` | Qwen vision-language model |
| `device` | `auto` | Computation device (cuda/cpu/auto) |
| `image_resolution` | `512` | Target image resolution |
| `temperature` | `0.7` | Text generation temperature |
| `max_new_tokens` | `2048` | Maximum response tokens |
| `seed` | `None` | Random seed for reproducibility |

## 🚀 Quick Start

1. **Setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your image paths and preferences
   ```

2. **Place your images:**
   ```bash
   mkdir -p input/person_images input/garment_images
   # Copy your person image to input/person_images/person1.jpg
   # Copy your garment image to input/garment_images/shirt1.jpg
   ```

3. **Run with defaults:**
   ```bash
   python main.py
   ```

4. **Check results:**
   ```bash
   ls outputs/
   # Generated: result.png, comparison.png, analysis.json
   ```

## 🔬 Examples

results = []
for person_img, desc in zip(person_images, descriptions):
    result = pipeline.intelligent_try_on(person_img, garment_image, desc)
    results.append(result)
```

## 🎯 Key Components

## 🔍 Examples

See `examples.py` for comprehensive usage examples:

```bash
# Run all examples
python examples.py

# Basic virtual try-on
python -c "from examples import example_basic_tryon; example_basic_tryon()"

# Batch processing
python -c "from examples import example_batch_processing; example_batch_processing()"

# Detailed analysis
python -c "from examples import example_detailed_analysis; example_detailed_analysis()"
```

## 📊 Analysis Output

The pipeline provides detailed Qwen3.5 analysis including:

- **Person assessment**: pose, current clothing, visible areas
- **Garment analysis**: style, color, type, fit characteristics  
- **Compatibility evaluation**: how well the garment fits the person
- **Placement instructions**: detailed guidance for optimal try-on
- **Style recommendations**: adjustments needed for natural appearance
- **Technical guidance**: lighting, shadows, composition details

Example analysis output:
```text
The person appears to be wearing casual attire and has a relaxed pose. 
The garment is a red button-up shirt with modern styling. 
Compatibility assessment: The shirt would fit well given the person's build.
Placement instructions: Position the shirt to replace the current upper garment...
[Detailed technical analysis continues...]
```

```bash
# Run all examples
python examples.py

# Or run specific examples in the file:
# - Basic virtual try-on
# - Batch processing
# - Detailed analysis
# - Custom configuration
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce image resolution or use CPU
config = QwenTryOnConfig(
    image_resolution=256,  # Smaller resolution
    device="cpu"           # Use CPU
)
```

**2. Person Not Detected**
- Ensure person is clearly visible and well-lit
- Try images with single person in frame
- Check that person matches the description

**3. Poor Results**
- Use high-quality input images (good lighting, clear clothing)
- Provide accurate person descriptions
- Experiment with different guidance scales

### Performance Optimization

```python
# For faster processing
config = QwenTryOnConfig(
    num_inference_steps=25,    # Fewer steps
    image_resolution=256,      # Lower resolution
    use_pose_control=False     # Skip pose guidance
)

# For higher quality
config = QwenTryOnConfig(
    num_inference_steps=75,    # More steps
    image_resolution=768,      # Higher resolution  
    guidance_scale=8.0         # Stronger guidance
)
```

## 📚 API Reference

### QwenVirtualTryOnPipeline

**`intelligent_try_on(person_image, garment_image, person_description, use_pose_control=True, return_analysis=False)`**

Main method for performing virtual try-on.

- `person_image`: Path or PIL Image of the person
- `garment_image`: Path or PIL Image of the garment  
- `person_description`: Text description of the target person
- `use_pose_control`: Whether to use pose guidance (default: True)
- `return_analysis`: Whether to return analysis data (default: False)

Returns: PIL Image (result), optionally with analysis Dict

### QwenVisionAnalyzer

**`analyze_person_image(person_image, person_description)`**

Analyzes person image using Qwen3.5's vision capabilities.

**`analyze_garment_image(garment_image)`**

Analyzes garment characteristics and style.

**`generate_tryon_instructions(person_analysis, garment_analysis, person_image, garment_image)`**

Generates detailed try-on instructions and compatibility assessment.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- **Qwen Team** for the excellent multimodal language model
- **Hugging Face** for Transformers and Diffusers libraries
- **Stability AI** for Stable Diffusion
- **ControlNet** team for pose-guided generation

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the examples in `examples.py`
3. Open an issue on GitHub with detailed information about your problem

---

**Happy Virtual Try-On with Qwen3.5! 👕✨**