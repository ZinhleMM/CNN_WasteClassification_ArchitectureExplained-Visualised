# CNN Visualization Toolkit for Waste Classification

This toolkit provides comprehensive visualizations of CNN architecture, layers, and transformations for waste classification models. Inspired by Figure 1.6 from "Deep Learning with Python" and tutorials from GeeksforGeeks and DataCamp.

## Overview

The toolkit includes four main visualization modules:

### 1. CNN Architecture Visualizer (`cnn_architecture_visualizer.py`)
- **Purpose**: Shows complete CNN architecture with layer flow
- **Features**:
  - Layer-by-layer architecture diagram
  - Data transformation visualization (Figure 1.6 style)
  - Convolution operation demonstrations
  - Pooling operation examples
- **Outputs**: Architecture diagrams, transformation flows, operation examples

### 2. Feature Map Extractor (`feature_map_extractor.py`)
- **Purpose**: Extracts and visualizes actual feature maps from trained models
- **Features**:
  - Real feature map extraction from model layers
  - Layer progression visualization
  - Multi-image comparison
  - Feature evolution analysis
- **Outputs**: Feature map grids, progression diagrams, comparison plots

### 3. Interactive CNN Analyzer (`interactive_cnn_analyzer.py`)
- **Purpose**: Provides detailed analysis of model architecture and complexity
- **Features**:
  - Layer complexity analysis
  - Parameter distribution visualization
  - Data flow analysis
  - Comprehensive model reports
- **Outputs**: Analysis charts, complexity metrics, detailed reports

### 4. Figure 1.6 Recreator (`figure_1_6_recreator.py`)
- **Purpose**: Recreates Figure 1.6 style visualization for waste classification
- **Features**:
  - Exact Figure 1.6 layout recreation
  - Layer-by-layer feature map display
  - Classification output visualization
  - Multiple image examples
- **Outputs**: Figure 1.6 style diagrams showing data representations

## Installation Requirements

```bash
# Core visualization libraries
pip install matplotlib seaborn numpy

# Image processing
pip install pillow

# Scientific computing
pip install scipy

# For model loading (optional)
pip install tensorflow
```

## Usage Examples

### Quick Start
```python
# Run complete demonstration
python cnn_visualization_demo.py

# Individual modules
from cnn_architecture_visualizer import CNNArchitectureVisualizer
from feature_map_extractor import FeatureMapExtractor
from interactive_cnn_analyzer import InteractiveCNNAnalyzer
from figure_1_6_recreator import Figure16Recreator

# Create visualizations
visualizer = CNNArchitectureVisualizer()
model_info = {'num_classes': 5, 'class_names': ['plastic', 'organic', 'metal', 'glass', 'paper']}
visualizer.create_complete_cnn_visualization(model_info)
```

### Architecture Analysis
```python
analyzer = InteractiveCNNAnalyzer('path/to/model.keras')
analyzer.create_comprehensive_analysis(sample_image_path='sample.png')
```

### Feature Map Extraction
```python
extractor = FeatureMapExtractor('path/to/model.keras')
feature_data = extractor.extract_feature_maps('image.png')
extractor.visualize_layer_progression(feature_data)
```

### Figure 1.6 Recreation
```python
recreator = Figure16Recreator('path/to/model.keras')
recreator.create_figure_1_6_visualization('waste_image.png', 'output.png')
```

## Visualization Types

### 1. Architecture Diagrams
- Complete model flow visualization
- Layer-by-layer breakdown
- Parameter count analysis
- Data transformation tracking

### 2. Feature Map Analysis
- Layer activation visualization
- Feature evolution through network
- Filter response patterns
- Spatial dimension changes

### 3. Figure 1.6 Style Recreations
- Original input image display
- Layer representation grids
- Feature map progressions
- Classification output bars

### 4. Operation Demonstrations
- Convolution filter examples
- Pooling operation illustrations
- Activation function effects
- Data flow animations

## Key Concepts Visualized

### Convolution Operations
- How filters detect features
- Edge detection examples
- Pattern recognition stages
- Feature map generation

### Pooling Operations
- Max pooling vs average pooling
- Spatial dimension reduction
- Feature preservation strategies
- Translation invariance

### Layer Transformations
- Input → Feature maps → Classification
- Increasing feature complexity
- Decreasing spatial resolution
- Abstract feature learning

### Data Flow
- End-to-end processing pipeline
- Layer-by-layer transformations
- Feature hierarchy development
- Classification decision process

## Model Compatibility

The toolkit works with:
- TensorFlow/Keras models
- Transfer learning architectures (MobileNetV2, ResNet, etc.)
- Custom CNN architectures
- Waste classification models

For models not available, the toolkit can simulate realistic visualizations.

## Output Files

Generated visualizations include:
- `01_cnn_architecture.png` - Complete architecture diagram
- `02_layer_transformations.png` - Figure 1.6 style progression
- `03_convolution_operations.png` - Filter operation examples
- `04_pooling_operations.png` - Pooling demonstrations
- `feature_maps_*.png` - Real feature map extractions
- `layer_progression_*.png` - Layer-by-layer analysis
- `model_analysis_report.json` - Detailed model metrics

## References

1. **Figure 1.6**: "Data representations learned by a digit-classification model" from "Deep Learning with Python" by François Chollet
2. **GeeksforGeeks**: [Introduction to Convolution Neural Network](https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/)
3. **DataCamp**: [Convolutional Neural Networks in Python](https://www.datacamp.com/tutorial/convolutional-neural-networks-python)
4. **Keras Documentation**: Loss functions and model analysis

## Customization

The toolkit can be adapted for different:
- Model architectures
- Image classification tasks
- Visualization styles
- Output formats

Each module includes extensive customization options for colors, layouts, and analysis depth.

## Technical Details

### CNN Architecture Support
- MobileNetV2-based transfer learning
- Custom classification heads
- Multiple layer types (Conv2D, Dense, Dropout, Pooling)
- Batch normalization and activation layers

### Visualization Features
- High-resolution output (300 DPI)
- Professional styling
- Color-coded layer types
- Detailed annotations and legends

### Analysis Capabilities
- Parameter counting and complexity metrics
- Memory usage estimation
- Layer efficiency analysis
- Feature map statistics

This toolkit provides a comprehensive understanding of how CNNs process waste classification images, making the "black box" of deep learning more interpretable and educational.
