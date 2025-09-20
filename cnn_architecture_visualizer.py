#!/usr/bin/env python3
"""
CNN Architecture Visualizer

This module provides comprehensive visualization of CNN architectures,
layer details, and data flow transformations. Inspired by Figure 1.6
showing data representations learned by neural networks.

References:
- GeeksforGeeks CNN Introduction
- DataCamp CNN Tutorial
- Deep Learning with Python by François Chollet

Author: CNN Visualization Tool
Date: 2025-08-03
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class CNNArchitectureVisualizer:
    """
    Visualize CNN architecture, layers, and transformations.

    This class creates publication-quality visualizations of CNN architectures
    showing data flow, layer transformations, and feature map evolution.
    """

    def __init__(self, figsize=(16, 10)):
        """Initialize the visualizer with figure settings."""
        self.figsize = figsize
        self.colors = {
            'input': '#E8F4FD',
            'conv': '#4A90E2',
            'pool': '#F5A623',
            'dense': '#7ED321',
            'output': '#D0021B',
            'activation': '#9013FE',
            'dropout': '#BD10E0'
        }

    def visualize_cnn_architecture(self, model_info: Dict, save_path: Optional[str] = None):
        """
        Create a detailed CNN architecture diagram showing layer flow.

        Args:
            model_info: Dictionary containing model architecture details
            save_path: Optional path to save the visualization
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Define architecture layers for waste classification CNN
        layers = [
            {'name': 'Input Image', 'type': 'input', 'shape': '224×224×3', 'description': 'RGB Waste Image'},
            {'name': 'MobileNetV2\nBase Model', 'type': 'conv', 'shape': '7×7×1280', 'description': 'Feature Extraction'},
            {'name': 'Global Average\nPooling2D', 'type': 'pool', 'shape': '1×1×1280', 'description': 'Spatial Reduction'},
            {'name': 'Dropout\n(0.2)', 'type': 'dropout', 'shape': '1280', 'description': 'Regularization'},
            {'name': 'Dense\n(128 units)', 'type': 'dense', 'shape': '128', 'description': 'Feature Combination'},
            {'name': 'Dropout\n(0.2)', 'type': 'dropout', 'shape': '128', 'description': 'Regularization'},
            {'name': 'Output\n(Softmax)', 'type': 'output', 'shape': f"{model_info.get('num_classes', 'N')}", 'description': 'Waste Classification'}
        ]

        # Calculate positions
        layer_width = 1.5
        layer_height = 1.0
        spacing = 2.5
        start_x = 1
        y_center = 3

        # Draw layers
        for i, layer in enumerate(layers):
            x = start_x + i * spacing

            # Draw layer box
            rect = patches.FancyBboxPatch(
                (x - layer_width/2, y_center - layer_height/2),
                layer_width, layer_height,
                boxstyle="round,pad=0.1",
                facecolor=self.colors[layer['type']],
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)

            # Add layer text
            ax.text(x, y_center + 0.1, layer['name'],
                   ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(x, y_center - 0.2, layer['shape'],
                   ha='center', va='center', fontsize=8, style='italic')
            ax.text(x, y_center - 0.7, layer['description'],
                   ha='center', va='center', fontsize=7, wrap=True)

            # Draw arrows between layers
            if i < len(layers) - 1:
                arrow = patches.FancyArrowPatch(
                    (x + layer_width/2, y_center),
                    (x + spacing - layer_width/2, y_center),
                    connectionstyle="arc3",
                    arrowstyle='->',
                    mutation_scale=20,
                    color='black',
                    linewidth=2
                )
                ax.add_patch(arrow)

        # Add title and labels
        ax.set_title('CNN Architecture for Waste Classification\nData Flow and Layer Transformations',
                    fontsize=16, fontweight='bold', pad=20)

        # Add legend
        legend_elements = [
            patches.Patch(facecolor=self.colors['input'], label='Input Layer'),
            patches.Patch(facecolor=self.colors['conv'], label='Convolutional Layers'),
            patches.Patch(facecolor=self.colors['pool'], label='Pooling Layers'),
            patches.Patch(facecolor=self.colors['dropout'], label='Dropout Layers'),
            patches.Patch(facecolor=self.colors['dense'], label='Dense Layers'),
            patches.Patch(facecolor=self.colors['output'], label='Output Layer')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        # Set axis properties
        ax.set_xlim(0, start_x + len(layers) * spacing)
        ax.set_ylim(1, 5)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ CNN architecture visualization saved to: {save_path}")

        plt.show()

    def visualize_layer_transformations(self, save_path: Optional[str] = None):
        """
        Create Figure 1.6 style visualization showing layer-by-layer transformations.

        This mimics the data representations learned by the digit classification model
        but adapted for waste classification CNN.
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Simulate layer representations (in practice, these would come from actual model)
        np.random.seed(42)

        # Original input (waste image)
        original_size = 64
        original_image = self._create_sample_waste_image(original_size)

        # Layer representations with decreasing spatial size but increasing channels
        layer_info = [
            {'name': 'Layer 1\nrepresentations', 'size': 32, 'channels': 4, 'description': 'Edge Detection'},
            {'name': 'Layer 2\nrepresentations', 'size': 16, 'channels': 8, 'description': 'Pattern Recognition'},
            {'name': 'Layer 3\nrepresentations', 'size': 8, 'channels': 16, 'description': 'Object Parts'},
            {'name': 'Layer 4\nrepresentations\n(final output)', 'size': 1, 'channels': 1, 'description': 'Classification'}
        ]

        # Position calculations
        input_x = 1
        layer_spacing = 3
        y_center = 4
        feature_map_spacing = 0.8

        # Draw original input
        ax.imshow(original_image, extent=[input_x-0.5, input_x+0.5, y_center-0.5, y_center+0.5],
                 cmap='gray', alpha=0.9)
        ax.text(input_x, y_center-1, 'Original\ninput', ha='center', va='center',
               fontsize=10, fontweight='bold')

        # Draw layers and their representations
        for i, layer in enumerate(layer_info):
            layer_x = input_x + (i + 1) * layer_spacing

            # Draw layer box
            if i < len(layer_info) - 1:  # Not the final output
                # Show sample feature maps
                num_samples = min(4, layer['channels'])
                for j in range(num_samples):
                    y_pos = y_center + (j - num_samples/2 + 0.5) * feature_map_spacing

                    # Generate synthetic feature map
                    feature_map = self._generate_feature_map(layer['size'], layer_type=i)

                    # Display feature map
                    extent = [layer_x - 0.3, layer_x + 0.3, y_pos - 0.3, y_pos + 0.3]
                    ax.imshow(feature_map, extent=extent, cmap='viridis', alpha=0.8)

                    # Add connecting lines from input/previous layer
                    if i == 0:
                        # Connect from original input
                        ax.plot([input_x + 0.5, layer_x - 0.3], [y_center, y_pos],
                               'k-', alpha=0.3, linewidth=1)
                    else:
                        # Connect from previous layer
                        prev_x = input_x + i * layer_spacing
                        ax.plot([prev_x + 0.3, layer_x - 0.3], [y_center, y_pos],
                               'k-', alpha=0.3, linewidth=1)
            else:
                # Final output - classification scores
                class_names = ['plastic', 'organic', 'metal', 'glass', 'paper']
                scores = np.random.dirichlet(np.ones(len(class_names)))

                # Draw classification bar
                bar_height = 2
                bar_width = 0.3
                for j, (name, score) in enumerate(zip(class_names, scores)):
                    y_pos = y_center + (j - len(class_names)/2 + 0.5) * 0.3
                    bar_length = score * bar_width

                    rect = patches.Rectangle((layer_x - bar_width/2, y_pos - 0.1),
                                           bar_length, 0.2,
                                           facecolor='black', alpha=score)
                    ax.add_patch(rect)

                    ax.text(layer_x + bar_width/2 + 0.1, y_pos, f'{name}\n{score:.2f}',
                           ha='left', va='center', fontsize=8)

            # Add layer label
            ax.text(layer_x, y_center - 2, layer['name'], ha='center', va='center',
                   fontsize=10, fontweight='bold')
            ax.text(layer_x, y_center - 2.5, layer['description'], ha='center', va='center',
                   fontsize=8, style='italic')

            # Draw layer representation box
            if i < len(layer_info) - 1:
                rect = patches.Rectangle((layer_x - 0.5, y_center - 1.8), 1, 0.4,
                                       facecolor='lightblue', alpha=0.3,
                                       edgecolor='blue', linewidth=1)
                ax.add_patch(rect)
                ax.text(layer_x, y_center - 1.6, f'Layer {i+1}', ha='center', va='center',
                       fontsize=9, fontweight='bold')

        # Add title
        ax.set_title('Data Representations Learned by Waste Classification CNN\n' +
                    'Feature Evolution Through Network Layers',
                    fontsize=14, fontweight='bold', pad=20)

        # Set axis properties
        ax.set_xlim(0, input_x + len(layer_info) * layer_spacing + 1)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Layer transformations visualization saved to: {save_path}")

        plt.show()

    def visualize_convolution_operation(self, save_path: Optional[str] = None):
        """
        Visualize how convolution operations work step by step.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Create sample input image (waste item)
        input_img = self._create_sample_waste_image(8)

        # Define different types of filters
        filters = {
            'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
            'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        }

        # Show original image
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Original Input\n(Waste Image)', fontweight='bold')
        axes[0, 0].axis('off')

        # Show filters and their outputs
        for i, (filter_name, kernel) in enumerate(filters.items()):
            # Show filter
            axes[0, i+1].imshow(kernel, cmap='RdBu', vmin=-1, vmax=1)
            axes[0, i+1].set_title(f'{filter_name}\nFilter/Kernel', fontweight='bold')
            axes[0, i+1].axis('off')

            # Add values to filter visualization
            for row in range(kernel.shape[0]):
                for col in range(kernel.shape[1]):
                    axes[0, i+1].text(col, row, f'{kernel[row, col]:.1f}',
                                     ha='center', va='center', fontweight='bold')

            # Apply convolution
            output = self._apply_convolution(input_img, kernel)

            # Show output
            axes[1, i+1].imshow(output, cmap='gray')
            axes[1, i+1].set_title(f'Output after\n{filter_name}', fontweight='bold')
            axes[1, i+1].axis('off')

        # Show convolution process illustration
        axes[1, 0].text(0.5, 0.5, 'Convolution\nOperation:\n\n' +
                       '1. Slide filter over image\n' +
                       '2. Element-wise multiply\n' +
                       '3. Sum all products\n' +
                       '4. Store result\n' +
                       '5. Move to next position',
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')

        plt.suptitle('CNN Convolution Operation Visualization\nHow Filters Extract Features',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Convolution operation visualization saved to: {save_path}")

        plt.show()

    def visualize_pooling_operations(self, save_path: Optional[str] = None):
        """
        Visualize pooling operations (max pooling and average pooling).
        """
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Create sample feature map
        feature_map = self._generate_feature_map(8, layer_type=1)

        # Show original feature map
        axes[0, 0].imshow(feature_map, cmap='viridis')
        axes[0, 0].set_title('Original Feature Map\n8×8', fontweight='bold')
        axes[0, 0].axis('off')

        # Add grid lines
        for i in range(9):
            axes[0, 0].axhline(i-0.5, color='white', linewidth=1, alpha=0.7)
            axes[0, 0].axvline(i-0.5, color='white', linewidth=1, alpha=0.7)

        # Max pooling
        max_pooled = self._max_pooling(feature_map, pool_size=2)
        axes[0, 1].imshow(max_pooled, cmap='viridis')
        axes[0, 1].set_title('Max Pooling\n4×4 (2×2 window)', fontweight='bold')
        axes[0, 1].axis('off')

        # Average pooling
        avg_pooled = self._average_pooling(feature_map, pool_size=2)
        axes[0, 2].imshow(avg_pooled, cmap='viridis')
        axes[0, 2].set_title('Average Pooling\n4×4 (2×2 window)', fontweight='bold')
        axes[0, 2].axis('off')

        # Show pooling window examples
        axes[1, 0].text(0.5, 0.5, 'Pooling Operation:\n\n' +
                       '• Reduces spatial dimensions\n' +
                       '• Provides translation invariance\n' +
                       '• Reduces computational load\n' +
                       '• Helps prevent overfitting',
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')

        # Max pooling explanation
        axes[1, 1].text(0.5, 0.5, 'Max Pooling:\n\n' +
                       'Takes maximum value\nfrom each window\n\n' +
                       'Preserves strongest\nactivations\n\n' +
                       'Good for detecting\nif feature is present',
                       ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        # Average pooling explanation
        axes[1, 2].text(0.5, 0.5, 'Average Pooling:\n\n' +
                       'Takes average value\nfrom each window\n\n' +
                       'Preserves overall\nintensity\n\n' +
                       'Good for background\ninformation',
                       ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        plt.suptitle('CNN Pooling Operations\nSpatial Dimension Reduction',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Pooling operations visualization saved to: {save_path}")

        plt.show()

    def create_complete_cnn_visualization(self, model_info: Dict, save_dir: str = "cnn_visualizations"):
        """
        Create a complete set of CNN visualizations and save them.

        Args:
            model_info: Dictionary containing model information
            save_dir: Directory to save all visualizations
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        print("Creating comprehensive CNN visualizations...")
        print("="*50)

        # 1. Architecture overview
        self.visualize_cnn_architecture(
            model_info,
            save_path / "01_cnn_architecture.png"
        )

        # 2. Layer transformations (Figure 1.6 style)
        self.visualize_layer_transformations(
            save_path / "02_layer_transformations.png"
        )

        # 3. Convolution operations
        self.visualize_convolution_operation(
            save_path / "03_convolution_operations.png"
        )

        # 4. Pooling operations
        self.visualize_pooling_operations(
            save_path / "04_pooling_operations.png"
        )

        print(f"\n✓ All visualizations saved to: {save_path}")
        print("Files created:")
        for file in sorted(save_path.glob("*.png")):
            print(f"  - {file.name}")

    # Helper methods
    def _create_sample_waste_image(self, size: int) -> np.ndarray:
        """Create a synthetic waste image for demonstration."""
        np.random.seed(42)
        img = np.zeros((size, size))

        # Add some geometric patterns to simulate waste objects
        center = size // 2

        # Add circular pattern (bottle cap, etc.)
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 < (size//4)**2
        img[mask] = 0.8

        # Add some linear features (edges, etc.)
        img[center-1:center+1, :] = 0.6
        img[:, center-1:center+1] = 0.4

        # Add noise
        noise = np.random.normal(0, 0.1, (size, size))
        img = np.clip(img + noise, 0, 1)

        return img

    def _generate_feature_map(self, size: int, layer_type: int) -> np.ndarray:
        """Generate synthetic feature maps for different layers."""
        np.random.seed(42 + layer_type)

        if layer_type == 0:  # Early layer - edge-like features
            feature_map = np.random.rand(size, size)
            # Add some edge-like patterns
            feature_map[:, size//2] += 0.5
            feature_map[size//2, :] += 0.3
        elif layer_type == 1:  # Middle layer - pattern combinations
            feature_map = np.random.rand(size, size) * 0.5
            # Add blob-like patterns
            center = size // 2
            y, x = np.ogrid[:size, :size]
            mask = (x - center)**2 + (y - center)**2 < (size//3)**2
            feature_map[mask] += 0.5
        else:  # Later layer - more abstract features
            feature_map = np.random.rand(size, size) * 0.3
            # Add more diffuse patterns
            feature_map += np.random.normal(0, 0.2, (size, size))

        return np.clip(feature_map, 0, 1)

    def _apply_convolution(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply convolution operation (simplified implementation)."""
        from scipy.signal import convolve2d
        return convolve2d(img, kernel, mode='valid')

    def _max_pooling(self, feature_map: np.ndarray, pool_size: int) -> np.ndarray:
        """Apply max pooling operation."""
        h, w = feature_map.shape
        new_h, new_w = h // pool_size, w // pool_size
        result = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                window = feature_map[i*pool_size:(i+1)*pool_size,
                                   j*pool_size:(j+1)*pool_size]
                result[i, j] = np.max(window)

        return result

    def _average_pooling(self, feature_map: np.ndarray, pool_size: int) -> np.ndarray:
        """Apply average pooling operation."""
        h, w = feature_map.shape
        new_h, new_w = h // pool_size, w // pool_size
        result = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                window = feature_map[i*pool_size:(i+1)*pool_size,
                                   j*pool_size:(j+1)*pool_size]
                result[i, j] = np.mean(window)

        return result

# Example usage and demo
if __name__ == "__main__":
    # Example model information (would come from your actual model)
    model_info = {
        'num_classes': 5,
        'class_names': ['plastic', 'organic', 'metal', 'glass', 'paper'],
        'input_shape': (224, 224, 3),
        'model_params': 2257000
    }

    # Create visualizer
    visualizer = CNNArchitectureVisualizer()

    # Create all visualizations
    visualizer.create_complete_cnn_visualization(model_info)
