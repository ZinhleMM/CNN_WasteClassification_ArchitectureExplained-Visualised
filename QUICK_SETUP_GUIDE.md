# Quick Setup for CNN Visualization Toolkit

## Prerequisites
```bash
# Install visualization libraries
pip install matplotlib seaborn numpy pillow scipy

# Optional: Install TensorFlow for model loading
pip install tensorflow
```

## Running Visualizations

### Option 1: Complete Demo (Recommended)
```bash
python cnn_visualization_demo.py
```
This runs all visualizations and creates a complete analysis.

### Option 2: Individual Modules
```python
# Architecture visualization
from cnn_architecture_visualizer import CNNArchitectureVisualizer
visualizer = CNNArchitectureVisualizer()
visualizer.create_complete_cnn_visualization(model_info)

# Feature extraction (requires model file)
from feature_map_extractor import FeatureMapExtractor
extractor = FeatureMapExtractor('uploads/waste_model.keras')
feature_data = extractor.extract_feature_maps('uploads/sample_image.png')

# Figure 1.6 recreation
from figure_1_6_recreator import Figure16Recreator
recreator = Figure16Recreator('uploads/waste_model.keras')
recreator.create_figure_1_6_visualization('uploads/sample_image.png')
```

### Option 3: Simplified Demo (No Dependencies)
```bash
python simple_cnn_demo.py
```
This shows concepts without requiring visualization libraries.

## Expected Output Structure
```
demo_output/
├── 01_architecture/
│   ├── 01_cnn_architecture.png
│   ├── 02_layer_transformations.png
│   ├── 03_convolution_operations.png
│   └── 04_pooling_operations.png
├── 02_feature_maps/
│   ├── feature_maps_1_*.png
│   ├── layer_progression_1_*.png
│   └── feature_comparison.png
├── 03_analysis/
│   ├── 01_model_architecture.png
│   ├── 02_layer_complexity.png
│   ├── 03_data_flow.png
│   └── 04_model_report.json
└── 04_figure_1_6/
    ├── figure_1_6_example_1_*.png
    ├── figure_1_6_example_2_*.png
    └── figure_1_6_example_3_*.png
```

## Troubleshooting
- If libraries aren't available: Run `simple_cnn_demo.py`
- If model file missing: Toolkit will simulate visualizations
- If images missing: Synthetic examples will be created
- For help: Check README_CNN_Visualization_Toolkit.md
