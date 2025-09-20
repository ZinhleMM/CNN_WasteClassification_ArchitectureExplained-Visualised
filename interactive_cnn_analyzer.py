#!/usr/bin/env python3
"""
Interactive CNN Layer Analyzer

This module provides detailed analysis and visualization of CNN model layers,
weights, filters, and transformations. Designed to work with the waste
classification model architecture.

References:
- Deep Learning with Python by François Chollet
- GeeksforGeeks CNN tutorials
- DataCamp CNN implementation guide

Author: Interactive CNN Analysis Tool
Date: 2025-08-03
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Handle TensorFlow/Keras imports gracefully
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Using simulated data.")
    TF_AVAILABLE = False

class InteractiveCNNAnalyzer:
    """
    Comprehensive CNN analysis and visualization tool.

    Provides detailed insights into model architecture, layer weights,
    filter patterns, and activation patterns.
    """

    def __init__(self, model_path: Optional[str] = None, class_names: Optional[List[str]] = None):
        """
        Initialize the CNN analyzer.

        Args:
            model_path: Path to trained model file
            class_names: List of class names for classification
        """
        self.model_path = model_path
        self.model = None
        self.class_names = class_names or ['plastic', 'organic', 'metal', 'glass', 'paper']
        self.layer_info = []

        if TF_AVAILABLE and model_path and Path(model_path).exists():
            self._load_and_analyze_model()
        else:
            self._create_synthetic_model_info()

    def _load_and_analyze_model(self):
        """Load model and extract detailed layer information."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded: {self.model_path}")
            self._extract_layer_info()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._create_synthetic_model_info()

    def _extract_layer_info(self):
        """Extract detailed information about each layer."""
        self.layer_info = []

        for i, layer in enumerate(self.model.layers):
            layer_data = {
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'trainable_params': layer.count_params() if hasattr(layer, 'count_params') else 0,
                'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
                'input_shape': layer.input_shape if hasattr(layer, 'input_shape') else None,
                'config': layer.get_config() if hasattr(layer, 'get_config') else {}
            }

            # Add layer-specific information
            if hasattr(layer, 'filters'):
                layer_data['filters'] = layer.filters
            if hasattr(layer, 'kernel_size'):
                layer_data['kernel_size'] = layer.kernel_size
            if hasattr(layer, 'strides'):
                layer_data['strides'] = layer.strides
            if hasattr(layer, 'padding'):
                layer_data['padding'] = layer.padding
            if hasattr(layer, 'activation'):
                layer_data['activation'] = str(layer.activation)
            if hasattr(layer, 'pool_size'):
                layer_data['pool_size'] = layer.pool_size
            if hasattr(layer, 'rate'):  # Dropout rate
                layer_data['dropout_rate'] = layer.rate

            self.layer_info.append(layer_data)

    def _create_synthetic_model_info(self):
        """Create synthetic model information for demonstration."""
        self.layer_info = [
            {
                'index': 0,
                'name': 'input_layer',
                'type': 'InputLayer',
                'trainable_params': 0,
                'output_shape': (None, 224, 224, 3),
                'input_shape': (None, 224, 224, 3),
                'config': {}
            },
            {
                'index': 1,
                'name': 'mobilenetv2_base',
                'type': 'Functional',
                'trainable_params': 2257984,
                'output_shape': (None, 7, 7, 1280),
                'input_shape': (None, 224, 224, 3),
                'config': {'frozen': True}
            },
            {
                'index': 2,
                'name': 'global_average_pooling2d',
                'type': 'GlobalAveragePooling2D',
                'trainable_params': 0,
                'output_shape': (None, 1280),
                'input_shape': (None, 7, 7, 1280),
                'config': {}
            },
            {
                'index': 3,
                'name': 'dropout',
                'type': 'Dropout',
                'trainable_params': 0,
                'output_shape': (None, 1280),
                'input_shape': (None, 1280),
                'dropout_rate': 0.2,
                'config': {}
            },
            {
                'index': 4,
                'name': 'dense',
                'type': 'Dense',
                'trainable_params': 164608,
                'output_shape': (None, 128),
                'input_shape': (None, 1280),
                'activation': 'relu',
                'config': {}
            },
            {
                'index': 5,
                'name': 'dropout_1',
                'type': 'Dropout',
                'trainable_params': 0,
                'output_shape': (None, 128),
                'input_shape': (None, 128),
                'dropout_rate': 0.2,
                'config': {}
            },
            {
                'index': 6,
                'name': 'classification_output',
                'type': 'Dense',
                'trainable_params': 645,
                'output_shape': (None, 5),
                'input_shape': (None, 128),
                'activation': 'softmax',
                'config': {}
            }
        ]

    def visualize_model_architecture(self, save_path: Optional[str] = None):
        """
        Create detailed architecture visualization with layer information.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

        # Left plot: Architecture flow
        self._plot_architecture_flow(ax1)

        # Right plot: Layer details table
        self._plot_layer_details_table(ax2)

        plt.suptitle('CNN Model Architecture Analysis\nWaste Classification Network',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Architecture analysis saved to: {save_path}")

        plt.show()

    def _plot_architecture_flow(self, ax):
        """Plot the architecture flow diagram."""
        # Color scheme for different layer types
        colors = {
            'InputLayer': '#E8F4FD',
            'Functional': '#4A90E2',
            'Conv2D': '#4A90E2',
            'MaxPooling2D': '#F5A623',
            'GlobalAveragePooling2D': '#F5A623',
            'Dense': '#7ED321',
            'Dropout': '#BD10E0',
            'Activation': '#9013FE'
        }

        y_positions = np.linspace(9, 1, len(self.layer_info))

        for i, (layer, y_pos) in enumerate(zip(self.layer_info, y_positions)):
            # Get layer color
            layer_color = colors.get(layer['type'], '#CCCCCC')

            # Draw layer box
            width = 3
            height = 0.8
            rect = plt.Rectangle((1, y_pos - height/2), width, height,
                               facecolor=layer_color, edgecolor='black',
                               linewidth=2, alpha=0.8)
            ax.add_patch(rect)

            # Add layer text
            layer_text = f"{layer['name']}\n({layer['type']})"
            if layer['output_shape']:
                shape_str = str(layer['output_shape']).replace('None, ', '')
                layer_text += f"\n{shape_str}"

            ax.text(2.5, y_pos, layer_text, ha='center', va='center',
                   fontsize=9, fontweight='bold')

            # Add parameter count
            if layer['trainable_params'] > 0:
                param_text = f"{layer['trainable_params']:,} params"
                ax.text(4.2, y_pos, param_text, ha='left', va='center',
                       fontsize=8, style='italic')

            # Draw arrows between layers
            if i < len(self.layer_info) - 1:
                ax.arrow(2.5, y_pos - height/2, 0, -0.4,
                        head_width=0.1, head_length=0.1,
                        fc='black', ec='black', alpha=0.7)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 10)
        ax.set_title('Model Architecture Flow', fontweight='bold', fontsize=12)
        ax.axis('off')

    def _plot_layer_details_table(self, ax):
        """Plot detailed layer information table."""
        # Prepare table data
        table_data = []
        headers = ['Layer', 'Type', 'Output Shape', 'Params', 'Details']

        for layer in self.layer_info:
            details = []
            if 'filters' in layer:
                details.append(f"Filters: {layer['filters']}")
            if 'kernel_size' in layer:
                details.append(f"Kernel: {layer['kernel_size']}")
            if 'activation' in layer:
                details.append(f"Activation: {layer['activation'].split('.')[-1] if '.' in str(layer['activation']) else layer['activation']}")
            if 'dropout_rate' in layer:
                details.append(f"Rate: {layer['dropout_rate']}")

            row = [
                layer['name'][:15] + ('...' if len(layer['name']) > 15 else ''),
                layer['type'],
                str(layer['output_shape']).replace('None, ', '') if layer['output_shape'] else 'N/A',
                f"{layer['trainable_params']:,}" if layer['trainable_params'] > 0 else '0',
                '\n'.join(details[:2])  # Limit to 2 details per cell
            ]
            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color alternate rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')

        ax.set_title('Layer Details Summary', fontweight='bold', fontsize=12)
        ax.axis('off')

    def analyze_layer_complexity(self, save_path: Optional[str] = None):
        """
        Analyze and visualize layer complexity metrics.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Extract metrics
        layer_names = [layer['name'][:10] + ('...' if len(layer['name']) > 10 else '')
                      for layer in self.layer_info]
        param_counts = [layer['trainable_params'] for layer in self.layer_info]

        # 1. Parameter count by layer
        ax1.bar(range(len(layer_names)), param_counts, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Trainable Parameters')
        ax1.set_title('Trainable Parameters by Layer')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')

        # Add value labels on bars
        for i, v in enumerate(param_counts):
            if v > 0:
                ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)

        # 2. Cumulative parameter count
        cumulative_params = np.cumsum(param_counts)
        ax2.plot(range(len(layer_names)), cumulative_params, 'o-', color='green', linewidth=2)
        ax2.fill_between(range(len(layer_names)), cumulative_params, alpha=0.3, color='green')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Cumulative Parameters')
        ax2.set_title('Cumulative Parameter Growth')
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # 3. Layer type distribution
        layer_types = [layer['type'] for layer in self.layer_info]
        type_counts = {}
        for layer_type in layer_types:
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1

        wedges, texts, autotexts = ax3.pie(type_counts.values(), labels=type_counts.keys(),
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Layer Type Distribution')

        # 4. Output shape evolution
        output_shapes = []
        shape_labels = []
        for i, layer in enumerate(self.layer_info):
            if layer['output_shape'] and len(layer['output_shape']) > 1:
                # Calculate total output size (excluding batch dimension)
                shape = layer['output_shape'][1:]  # Remove batch dimension
                if all(isinstance(x, int) for x in shape):
                    total_size = np.prod(shape)
                    output_shapes.append(total_size)
                    shape_labels.append(f"Layer {i}")
                else:
                    output_shapes.append(0)
                    shape_labels.append(f"Layer {i}")

        if output_shapes:
            ax4.semilogy(range(len(output_shapes)), output_shapes, 'o-', color='red', linewidth=2)
            ax4.set_xlabel('Layer Index')
            ax4.set_ylabel('Output Size (log scale)')
            ax4.set_title('Output Tensor Size Evolution')
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(range(len(output_shapes)))
            ax4.set_xticklabels([f"L{i}" for i in range(len(output_shapes))])

        plt.suptitle('CNN Layer Complexity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Layer complexity analysis saved to: {save_path}")

        plt.show()

    def visualize_data_flow(self, sample_image_path: Optional[str] = None, save_path: Optional[str] = None):
        """
        Visualize how data flows through the network with actual shapes and sizes.
        """
        fig, ax = plt.subplots(figsize=(16, 10))

        # Calculate positions for data flow visualization
        x_positions = np.linspace(1, 15, len(self.layer_info))
        y_center = 5

        # Track tensor dimensions through the network
        for i, (layer, x_pos) in enumerate(zip(self.layer_info, x_positions)):
            # Get output shape
            if layer['output_shape']:
                shape = layer['output_shape'][1:]  # Remove batch dimension
                shape_text = f"Shape: {shape}"

                # Calculate relative size for visualization
                if all(isinstance(x, int) for x in shape):
                    total_size = np.prod(shape)
                    # Normalize size for visualization (log scale)
                    normalized_size = max(0.3, min(2.0, np.log10(total_size + 1) / 4))
                else:
                    normalized_size = 0.5
            else:
                shape_text = "Shape: N/A"
                normalized_size = 0.5

            # Choose color based on layer type
            color_map = {
                'InputLayer': 'lightblue',
                'Functional': 'lightgreen',
                'Conv2D': 'lightgreen',
                'GlobalAveragePooling2D': 'orange',
                'Dense': 'lightcoral',
                'Dropout': 'plum'
            }
            color = color_map.get(layer['type'], 'lightgray')

            # Draw data tensor representation
            width = normalized_size
            height = normalized_size * 0.8

            rect = plt.Rectangle((x_pos - width/2, y_center - height/2),
                               width, height, facecolor=color,
                               edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(rect)

            # Add layer information
            layer_text = f"{layer['name']}\n{layer['type']}\n{shape_text}"
            ax.text(x_pos, y_center, layer_text, ha='center', va='center',
                   fontsize=8, fontweight='bold', wrap=True)

            # Add transformation arrows
            if i < len(self.layer_info) - 1:
                next_x = x_positions[i + 1]
                arrow = plt.Arrow(x_pos + width/2, y_center,
                                next_x - x_pos - width, 0,
                                width=0.2, color='darkblue', alpha=0.7)
                ax.add_patch(arrow)

                # Add transformation description
                transform_text = self._get_transformation_description(layer, self.layer_info[i + 1])
                ax.text((x_pos + next_x) / 2, y_center + 1.5, transform_text,
                       ha='center', va='center', fontsize=7,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7))

        # Add input image representation
        if sample_image_path and Path(sample_image_path).exists():
            try:
                from PIL import Image
                img = Image.open(sample_image_path)
                img = img.resize((50, 50))
                img_array = np.array(img)

                # Display small version of input image
                ax.imshow(img_array, extent=[0.2, 0.8, y_center-0.3, y_center+0.3])
                ax.text(0.5, y_center-0.8, 'Input\nImage', ha='center', va='center',
                       fontsize=10, fontweight='bold')
            except Exception as e:
                print(f"Could not load sample image: {e}")

        # Formatting
        ax.set_xlim(0, 16)
        ax.set_ylim(2, 8)
        ax.set_title('CNN Data Flow Visualization\nTensor Shape Transformations Through Network',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

        # Add legend
        legend_elements = []
        for layer_type, color in {
            'Input': 'lightblue',
            'Convolution': 'lightgreen',
            'Pooling': 'orange',
            'Dense': 'lightcoral',
            'Dropout': 'plum'
        }.items():
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color,
                                               edgecolor='black', label=layer_type))

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Data flow visualization saved to: {save_path}")

        plt.show()

    def _get_transformation_description(self, current_layer: Dict, next_layer: Dict) -> str:
        """Get description of transformation between layers."""
        current_type = current_layer['type']
        next_type = next_layer['type']

        transformations = {
            ('InputLayer', 'Functional'): 'Feature\nExtraction',
            ('Functional', 'GlobalAveragePooling2D'): 'Spatial\nAggregation',
            ('GlobalAveragePooling2D', 'Dropout'): 'Flatten &\nRegularize',
            ('Dropout', 'Dense'): 'Feature\nCombination',
            ('Dense', 'Dropout'): 'Regularize',
            ('Dense', 'Dense'): 'Classification'
        }

        return transformations.get((current_type, next_type), 'Transform')

    def generate_model_summary_report(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive model analysis report.
        """
        # Calculate summary statistics
        total_params = sum(layer['trainable_params'] for layer in self.layer_info)
        trainable_layers = sum(1 for layer in self.layer_info if layer['trainable_params'] > 0)

        # Layer type analysis
        layer_types = [layer['type'] for layer in self.layer_info]
        type_counts = {}
        for layer_type in layer_types:
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1

        # Memory usage estimation (rough)
        total_memory_mb = 0
        for layer in self.layer_info:
            if layer['output_shape'] and len(layer['output_shape']) > 1:
                shape = layer['output_shape'][1:]
                if all(isinstance(x, int) for x in shape):
                    # Assume float32 (4 bytes per parameter)
                    layer_memory = np.prod(shape) * 4 / (1024 * 1024)
                    total_memory_mb += layer_memory

        report = {
            'model_summary': {
                'total_layers': len(self.layer_info),
                'trainable_layers': trainable_layers,
                'total_parameters': total_params,
                'estimated_memory_mb': round(total_memory_mb, 2),
                'model_size_mb': round(total_params * 4 / (1024 * 1024), 2)  # Rough estimate
            },
            'layer_distribution': type_counts,
            'complexity_metrics': {
                'parameters_per_layer': round(total_params / len(self.layer_info), 0),
                'max_layer_params': max(layer['trainable_params'] for layer in self.layer_info),
                'model_depth': len(self.layer_info)
            },
            'architecture_details': self.layer_info
        }

        # Print report
        print("="*60)
        print("CNN MODEL ANALYSIS REPORT")
        print("="*60)
        print(f"Model Depth: {report['model_summary']['total_layers']} layers")
        print(f"Trainable Layers: {report['model_summary']['trainable_layers']}")
        print(f"Total Parameters: {report['model_summary']['total_parameters']:,}")
        print(f"Estimated Model Size: {report['model_summary']['model_size_mb']} MB")
        print(f"Estimated Memory Usage: {report['model_summary']['estimated_memory_mb']} MB")

        print("\nLayer Type Distribution:")
        for layer_type, count in sorted(type_counts.items()):
            print(f"  {layer_type}: {count}")

        print("\nComplexity Metrics:")
        print(f"  Average params/layer: {report['complexity_metrics']['parameters_per_layer']:,.0f}")
        print(f"  Most complex layer: {report['complexity_metrics']['max_layer_params']:,} params")

        # Save detailed report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n✓ Detailed report saved to: {save_path}")

        return report

    def create_comprehensive_analysis(self, sample_image_path: Optional[str] = None,
                                    save_dir: str = "cnn_analysis"):
        """
        Create complete CNN analysis with all visualizations.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        print("Creating comprehensive CNN analysis...")
        print("="*50)

        # 1. Architecture visualization
        self.visualize_model_architecture(save_path / "01_model_architecture.png")

        # 2. Layer complexity analysis
        self.analyze_layer_complexity(save_path / "02_layer_complexity.png")

        # 3. Data flow visualization
        self.visualize_data_flow(sample_image_path, save_path / "03_data_flow.png")

        # 4. Generate summary report
        self.generate_model_summary_report(save_path / "04_model_report.json")

        print(f"\n✓ Complete analysis saved to: {save_path}")
        print("Files created:")
        for file in sorted(save_path.glob("*")):
            print(f"  - {file.name}")

# Example usage and demo
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = InteractiveCNNAnalyzer()

    # Run comprehensive analysis
    sample_image = None
    if Path("uploads/sample_image.png").exists():
        sample_image = "uploads/sample_image.png"
    elif Path("uploads/Batch_Directory_1_-_Classifier_Test_Image_1.png").exists():
        sample_image = "uploads/Batch_Directory_1_-_Classifier_Test_Image_1.png"

    # Create analysis
    analyzer.create_comprehensive_analysis(sample_image_path=sample_image)
