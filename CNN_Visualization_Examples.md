# CNN Visualization Usage Examples

## Example 1: Complete Architecture Analysis

```python
from cnn_architecture_visualizer import CNNArchitectureVisualizer

# Initialize visualizer
visualizer = CNNArchitectureVisualizer()

# Model information
model_info = {
    'num_classes': 5,
    'class_names': ['plastic', 'organic', 'metal', 'glass', 'paper'],
    'input_shape': (224, 224, 3),
    'model_params': 2422000
}

# Create all visualizations
visualizer.create_complete_cnn_visualization(model_info, save_dir="visualizations")
```

## Example 2: Feature Map Extraction

```python
from feature_map_extractor import FeatureMapExtractor

# Load model and extract features
extractor = FeatureMapExtractor("waste_model.keras")
feature_data = extractor.extract_feature_maps("sample_image.png")

# Visualize feature maps
extractor.visualize_feature_maps(feature_data, save_path="feature_maps.png")
extractor.visualize_layer_progression(feature_data, save_path="progression.png")

# Compare multiple images
image_paths = ["image1.png", "image2.png", "image3.png"]
extractor.compare_multiple_images(image_paths, save_path="comparison.png")
```

## Example 3: Interactive Model Analysis

```python
from interactive_cnn_analyzer import InteractiveCNNAnalyzer

# Analyze model architecture
analyzer = InteractiveCNNAnalyzer("waste_model.keras", class_names)

# Create comprehensive analysis
analyzer.visualize_model_architecture(save_path="architecture.png")
analyzer.analyze_layer_complexity(save_path="complexity.png")
analyzer.visualize_data_flow(sample_image_path="sample.png", save_path="dataflow.png")

# Generate detailed report
report = analyzer.generate_model_summary_report(save_path="model_report.json")
```

## Example 4: Figure 1.6 Recreation

```python
from figure_1_6_recreator import Figure16Recreator

# Create Figure 1.6 style visualization
recreator = Figure16Recreator("waste_model.keras")

# Single image visualization
recreator.create_figure_1_6_visualization(
    "waste_image.png",
    save_path="figure_1_6_recreation.png"
)

# Multiple image examples
image_list = ["plastic_bottle.png", "organic_waste.png", "metal_can.png"]
recreator.create_multiple_examples(image_list, save_dir="figure_1_6_examples")
```

## Example 5: Batch Processing

```python
import os
from pathlib import Path

# Process all images in a directory
image_dir = Path("test_images")
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# Initialize all tools
visualizer = CNNArchitectureVisualizer()
extractor = FeatureMapExtractor("model.keras")
analyzer = InteractiveCNNAnalyzer("model.keras")
recreator = Figure16Recreator("model.keras")

# Process each image
for img_path in image_dir.glob("*.png"):
    print(f"Processing {img_path.name}...")

    # Extract features
    features = extractor.extract_feature_maps(str(img_path))

    # Create visualizations
    extractor.visualize_layer_progression(
        features,
        save_path=str(output_dir / f"progression_{img_path.stem}.png")
    )

    recreator.create_figure_1_6_visualization(
        str(img_path),
        save_path=str(output_dir / f"figure_1_6_{img_path.stem}.png")
    )
```

## Example 6: Custom Visualization Settings

```python
# Customize visualization appearance
visualizer = CNNArchitectureVisualizer(figsize=(20, 12))

# Custom colors for different layer types
custom_colors = {
    'input': '#E8F4FD',
    'conv': '#4A90E2',
    'pool': '#F5A623',
    'dense': '#7ED321',
    'output': '#D0021B'
}
visualizer.colors = custom_colors

# Create architecture with custom settings
visualizer.visualize_cnn_architecture(model_info, save_path="custom_arch.png")
```

## Example 7: Model Comparison

```python
# Compare different models
models = [
    {"path": "model_v1.keras", "name": "Version 1"},
    {"path": "model_v2.keras", "name": "Version 2"},
    {"path": "model_v3.keras", "name": "Version 3"}
]

for model in models:
    analyzer = InteractiveCNNAnalyzer(model["path"])

    # Generate comparison reports
    report = analyzer.generate_model_summary_report(
        save_path=f"report_{model['name'].lower().replace(' ', '_')}.json"
    )

    print(f"{model['name']}: {report['model_summary']['total_parameters']:,} parameters")
```

## Example 8: Educational Demonstration

```python
# Create educational materials
def create_cnn_tutorial():
    visualizer = CNNArchitectureVisualizer()

    # 1. Show basic CNN concepts
    visualizer.visualize_convolution_operation(save_path="01_convolution.png")
    visualizer.visualize_pooling_operations(save_path="02_pooling.png")

    # 2. Show architecture progression
    simple_model = {
        'num_classes': 3,
        'class_names': ['plastic', 'organic', 'other']
    }
    visualizer.visualize_cnn_architecture(simple_model, save_path="03_architecture.png")

    # 3. Show layer transformations
    visualizer.visualize_layer_transformations(save_path="04_transformations.png")

create_cnn_tutorial()
```

## Example 9: Research Analysis

```python
# Detailed research analysis
def analyze_model_for_research(model_path, test_images):
    analyzer = InteractiveCNNAnalyzer(model_path)
    extractor = FeatureMapExtractor(model_path)

    # Model complexity analysis
    analyzer.analyze_layer_complexity(save_path="research_complexity.png")

    # Feature analysis across multiple images
    all_features = []
    for img_path in test_images:
        features = extractor.extract_feature_maps(img_path)
        all_features.append(features)

    # Compare feature activations
    extractor.compare_multiple_images(test_images, save_path="research_comparison.png")

    # Generate detailed report
    report = analyzer.generate_model_summary_report(save_path="research_report.json")

    return report

# Usage
test_images = ["test1.png", "test2.png", "test3.png"]
research_report = analyze_model_for_research("research_model.keras", test_images)
```

## Example 10: Integration with Training Pipeline

```python
# Integrate with model training
class TrainingVisualizer:
    def __init__(self, model_path):
        self.analyzer = InteractiveCNNAnalyzer(model_path)
        self.extractor = FeatureMapExtractor(model_path)

    def analyze_epoch(self, epoch, validation_images):
        # Create analysis for current epoch
        save_dir = f"epoch_{epoch}_analysis"
        Path(save_dir).mkdir(exist_ok=True)

        # Architecture analysis
        self.analyzer.create_comprehensive_analysis(
            sample_image_path=validation_images[0],
            save_dir=save_dir
        )

        # Feature evolution analysis
        for i, img in enumerate(validation_images[:3]):
            features = self.extractor.extract_feature_maps(img)
            self.extractor.visualize_layer_progression(
                features,
                save_path=f"{save_dir}/progression_img_{i+1}.png"
            )

# Usage during training
visualizer = TrainingVisualizer("current_model.keras")
validation_imgs = ["val1.png", "val2.png", "val3.png"]

# Call after each epoch
for epoch in range(10):
    # ... training code ...
    visualizer.analyze_epoch(epoch, validation_imgs)
```

These examples demonstrate the versatility and power of the CNN visualization toolkit for understanding, analyzing, and presenting neural network architectures and their learned representations.
