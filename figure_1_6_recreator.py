#!/usr/bin/env python3
"""
Figure 1.6 Style CNN Visualization Recreator

This module specifically recreates the "Data representations learned by a
digit-classification model" figure (Figure 1.6) but adapted for the waste
classification CNN model. Shows how input images are transformed through
successive layers with actual feature map visualizations.

References:
- Figure 1.6: Data representations learned by a digit-classification model
- Deep Learning with Python by François Chollet
- GeeksforGeeks CNN Introduction
- DataCamp CNN Tutorial

Author: Figure 1.6 Recreator for Waste Classification
Date: 2025-08-03
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Handle image processing imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Handle TensorFlow imports gracefully
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class Figure16Recreator:
    """
    Recreate Figure 1.6 style visualization for waste classification CNN.

    Shows the progression of data representations from input image through
    successive layers, ending with classification output.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Figure 1.6 recreator.

        Args:
            model_path: Path to trained CNN model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Plastic', 'Organic', 'Metal', 'Glass', 'Paper']

        if TF_AVAILABLE and model_path and Path(model_path).exists():
            self._load_model()

    def _load_model(self):
        """Load the trained model for feature extraction."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded for Figure 1.6 recreation: {self.model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Will use simulated feature representations.")

    def create_figure_1_6_visualization(self, input_image_path: str,
                                      save_path: Optional[str] = None):
        """
        Create the main Figure 1.6 style visualization.

        Args:
            input_image_path: Path to input waste image
            save_path: Optional path to save the visualization
        """
        fig, ax = plt.subplots(figsize=(16, 10))

        # Load and prepare input image
        original_image = self._load_input_image(input_image_path)

        # Extract or simulate layer representations
        layer_representations = self._extract_layer_representations(original_image)

        # Create the visualization layout
        self._create_figure_layout(ax, original_image, layer_representations)

        # Set title and formatting
        ax.set_title('Data Representations Learned by Waste Classification CNN\n' +
                    'Feature Evolution from Input to Classification',
                    fontsize=16, fontweight='bold', pad=30)

        ax.set_xlim(0, 18)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure 1.6 recreation saved to: {save_path}")

        plt.show()

    def _load_input_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess input image."""
        if PIL_AVAILABLE and Path(image_path).exists():
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                return img_array
            except Exception as e:
                print(f"Error loading image: {e}")

        # Create synthetic waste image if loading fails
        return self._create_synthetic_waste_image()

    def _create_synthetic_waste_image(self) -> np.ndarray:
        """Create a synthetic waste image that looks realistic."""
        np.random.seed(42)

        # Create base image
        img = np.zeros((224, 224, 3))

        # Add plastic bottle shape
        center_y, center_x = 112, 112

        # Bottle body (rectangular)
        body_top, body_bottom = 60, 160
        body_left, body_right = 80, 144
        img[body_top:body_bottom, body_left:body_right] = [0.3, 0.6, 0.8]  # Blue plastic

        # Bottle cap (circular)
        y, x = np.ogrid[:224, :224]
        cap_mask = (x - center_x)**2 + (y - 50)**2 < 20**2
        img[cap_mask] = [0.8, 0.2, 0.2]  # Red cap

        # Add label area
        label_top, label_bottom = 90, 130
        label_left, label_right = 85, 139
        img[label_top:label_bottom, label_left:label_right] = [0.9, 0.9, 0.9]  # White label

        # Add some texture and noise
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)

        # Add some background
        background_mask = np.sum(img, axis=2) < 0.1
        img[background_mask] = [0.95, 0.95, 0.9]  # Light background

        return img

    def _extract_layer_representations(self, input_image: np.ndarray) -> List[Dict]:
        """Extract or simulate layer representations."""
        if TF_AVAILABLE and self.model:
            return self._extract_real_representations(input_image)
        else:
            return self._simulate_layer_representations(input_image)

    def _extract_real_representations(self, input_image: np.ndarray) -> List[Dict]:
        """Extract actual feature representations from the model."""
        try:
            # Prepare input
            img_batch = np.expand_dims(input_image, axis=0)

            # Create intermediate models to extract feature maps
            layer_outputs = []
            layer_names = []

            # Get outputs from key layers
            for i, layer in enumerate(self.model.layers):
                if i == 0:  # Skip input layer
                    continue

                # For transfer learning models, get base model outputs
                if hasattr(layer, 'layers') and len(layer.layers) > 10:
                    # Sample some intermediate layers from base model
                    sample_indices = [len(layer.layers)//4, len(layer.layers)//2,
                                    3*len(layer.layers)//4, len(layer.layers)-1]
                    for idx in sample_indices:
                        if idx < len(layer.layers):
                            try:
                                intermediate_model = keras.Model(
                                    inputs=self.model.input,
                                    outputs=layer.layers[idx].output
                                )
                                output = intermediate_model.predict(img_batch, verbose=0)
                                if len(output.shape) == 4:  # Conv layer
                                    layer_outputs.append(output)
                                    layer_names.append(f"Base_Layer_{idx}")
                                    if len(layer_outputs) >= 3:  # Limit to 3 base layers
                                        break
                            except:
                                continue
                    break
                else:
                    # Regular layers
                    if hasattr(layer, 'output'):
                        try:
                            intermediate_model = keras.Model(
                                inputs=self.model.input,
                                outputs=layer.output
                            )
                            output = intermediate_model.predict(img_batch, verbose=0)
                            if len(output.shape) == 4:  # Conv layer
                                layer_outputs.append(output)
                                layer_names.append(layer.name)
                        except:
                            continue

            # Convert to standardized format
            representations = []
            for i, (output, name) in enumerate(zip(layer_outputs, layer_names)):
                rep = {
                    'name': f'Layer {i+1}\nrepresentations',
                    'description': self._get_layer_description(i),
                    'feature_maps': output[0],  # Remove batch dimension
                    'shape': output.shape[1:],
                    'type': 'conv'
                }
                representations.append(rep)

            return representations[:3]  # Limit to 3 intermediate layers

        except Exception as e:
            print(f"Error extracting real representations: {e}")
            return self._simulate_layer_representations(input_image)

    def _simulate_layer_representations(self, input_image: np.ndarray) -> List[Dict]:
        """Simulate realistic layer representations."""
        np.random.seed(42)

        representations = []

        # Layer 1: Edge detection and basic features
        layer1_maps = []
        for i in range(8):  # 8 feature maps
            # Simulate edge detection
            feature_map = np.random.rand(56, 56) * 0.3

            # Add edge-like patterns
            if i % 2 == 0:
                # Vertical edges
                feature_map[:, 28:30] += 0.6
            else:
                # Horizontal edges
                feature_map[28:30, :] += 0.6

            # Add some blob patterns
            center_y, center_x = 28, 28
            y, x = np.ogrid[:56, :56]
            blob_mask = (x - center_x)**2 + (y - center_y)**2 < 10**2
            feature_map[blob_mask] += 0.4

            feature_map = np.clip(feature_map, 0, 1)
            layer1_maps.append(feature_map)

        layer1_maps = np.stack(layer1_maps, axis=-1)
        representations.append({
            'name': 'Layer 1\nrepresentations',
            'description': 'Edge Detection\nBasic Features',
            'feature_maps': layer1_maps,
            'shape': layer1_maps.shape,
            'type': 'conv'
        })

        # Layer 2: Pattern combinations
        layer2_maps = []
        for i in range(6):  # 6 feature maps
            feature_map = np.random.rand(28, 28) * 0.2

            # Add combined patterns
            if i < 2:
                # Corner patterns
                feature_map[:14, :14] += 0.7
            elif i < 4:
                # Center patterns
                center_mask = (np.arange(28)[:, None] - 14)**2 + (np.arange(28) - 14)**2 < 8**2
                feature_map[center_mask] += 0.8
            else:
                # Complex patterns
                feature_map[::4, ::4] += 0.6

            feature_map = np.clip(feature_map, 0, 1)
            layer2_maps.append(feature_map)

        layer2_maps = np.stack(layer2_maps, axis=-1)
        representations.append({
            'name': 'Layer 2\nrepresentations',
            'description': 'Pattern Combination\nShape Detection',
            'feature_maps': layer2_maps,
            'shape': layer2_maps.shape,
            'type': 'conv'
        })

        # Layer 3: High-level features
        layer3_maps = []
        for i in range(4):  # 4 feature maps
            feature_map = np.random.rand(14, 14) * 0.1

            # Add object-like patterns
            if i == 0:
                # Circular object (bottle cap, can top)
                center_mask = (np.arange(14)[:, None] - 7)**2 + (np.arange(14) - 7)**2 < 4**2
                feature_map[center_mask] += 0.9
            elif i == 1:
                # Rectangular object (bottle body, box)
                feature_map[4:10, 3:11] += 0.8
            else:
                # Abstract patterns
                feature_map += np.random.rand(14, 14) * 0.5

            feature_map = np.clip(feature_map, 0, 1)
            layer3_maps.append(feature_map)

        layer3_maps = np.stack(layer3_maps, axis=-1)
        representations.append({
            'name': 'Layer 3\nrepresentations',
            'description': 'Object Parts\nHigh-level Features',
            'feature_maps': layer3_maps,
            'shape': layer3_maps.shape,
            'type': 'conv'
        })

        return representations

    def _create_figure_layout(self, ax, original_image: np.ndarray,
                            layer_representations: List[Dict]):
        """Create the main figure layout matching Figure 1.6 style."""

        # Position parameters
        input_x = 2
        layer_x_positions = [6, 10, 14]
        output_x = 17
        y_center = 5

        # 1. Display original input image
        ax.imshow(original_image, extent=[input_x-1, input_x+1, y_center-1, y_center+1])

        # Add input label
        ax.text(input_x, y_center-2, 'Original\ninput', ha='center', va='center',
               fontsize=12, fontweight='bold')

        # 2. Display layer representations
        for layer_idx, (layer_rep, layer_x) in enumerate(zip(layer_representations, layer_x_positions)):
            self._draw_layer_representation(ax, layer_rep, layer_x, y_center, layer_idx + 1)

            # Draw connecting lines from previous layer/input
            prev_x = input_x if layer_idx == 0 else layer_x_positions[layer_idx - 1]
            self._draw_connections(ax, prev_x, layer_x, y_center, layer_rep['feature_maps'].shape[-1])

        # 3. Display final classification output
        self._draw_classification_output(ax, output_x, y_center)

        # Connect last layer to output
        if layer_representations:
            last_layer_x = layer_x_positions[-1]
            self._draw_connections(ax, last_layer_x, output_x, y_center, len(self.class_names))

        # 4. Add layer labels and descriptions
        self._add_layer_labels(ax, layer_x_positions, layer_representations)

    def _draw_layer_representation(self, ax, layer_rep: Dict, x_pos: float,
                                 y_center: float, layer_num: int):
        """Draw feature maps for a single layer."""
        feature_maps = layer_rep['feature_maps']
        num_maps = min(4, feature_maps.shape[-1])  # Show max 4 feature maps

        # Calculate positions for feature maps
        map_spacing = 1.2
        start_y = y_center + (num_maps - 1) * map_spacing / 2

        for i in range(num_maps):
            y_pos = start_y - i * map_spacing

            # Get feature map
            feature_map = feature_maps[:, :, i]

            # Normalize for display
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

            # Display feature map
            extent = [x_pos - 0.6, x_pos + 0.6, y_pos - 0.5, y_pos + 0.5]
            ax.imshow(feature_map, extent=extent, cmap='viridis', alpha=0.9)

            # Add border
            rect = patches.Rectangle((x_pos - 0.6, y_pos - 0.5), 1.2, 1.0,
                                   linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    def _draw_connections(self, ax, start_x: float, end_x: float, y_center: float, num_connections: int):
        """Draw connection lines between layers."""
        # Create fan-out/fan-in connections
        connection_spacing = 0.8
        start_y = y_center + (num_connections - 1) * connection_spacing / 2

        for i in range(min(4, num_connections)):  # Limit visual connections
            y_start = start_y - i * connection_spacing
            y_end = y_center + (3 - i) * connection_spacing / 3 - connection_spacing * 1.5

            # Draw connection line
            ax.plot([start_x + 1, end_x - 0.6], [y_start, y_end],
                   'k-', alpha=0.3, linewidth=0.8)

    def _draw_classification_output(self, ax, x_pos: float, y_center: float):
        """Draw the final classification output."""
        # Simulate classification scores
        np.random.seed(42)
        scores = np.random.dirichlet(np.ones(len(self.class_names)))
        sorted_indices = np.argsort(scores)[::-1]

        # Draw classification bars
        bar_spacing = 0.4
        start_y = y_center + (len(self.class_names) - 1) * bar_spacing / 2

        for i, class_idx in enumerate(sorted_indices):
            y_pos = start_y - i * bar_spacing
            score = scores[class_idx]

            # Draw confidence bar
            bar_width = score * 0.8
            rect = patches.Rectangle((x_pos - 0.4, y_pos - 0.1), bar_width, 0.2,
                                   facecolor='black', alpha=score, edgecolor='black')
            ax.add_patch(rect)

            # Add class label and score
            ax.text(x_pos + 0.5, y_pos, f'{self.class_names[class_idx]}\n{score:.3f}',
                   ha='left', va='center', fontsize=9)

        # Add output label
        ax.text(x_pos, y_center - 2.5, 'Layer 4\nrepresentations\n(final output)',
               ha='center', va='center', fontsize=10, fontweight='bold')

    def _add_layer_labels(self, ax, x_positions: List[float], layer_representations: List[Dict]):
        """Add layer labels and descriptions."""
        for i, (x_pos, layer_rep) in enumerate(zip(x_positions, layer_representations)):
            # Add layer name
            ax.text(x_pos, 1.5, layer_rep['name'], ha='center', va='center',
                   fontsize=11, fontweight='bold')

            # Add description
            ax.text(x_pos, 0.8, layer_rep['description'], ha='center', va='center',
                   fontsize=9, style='italic')

            # Add shape information
            shape_text = f"Shape: {layer_rep['shape'][:2]}"
            ax.text(x_pos, 0.3, shape_text, ha='center', va='center',
                   fontsize=8, alpha=0.7)

    def _get_layer_description(self, layer_index: int) -> str:
        """Get description for layer based on its position in network."""
        descriptions = [
            'Edge Detection\nBasic Features',
            'Pattern Combination\nShape Detection',
            'Object Parts\nHigh-level Features',
            'Semantic Features\nObject Recognition'
        ]

        return descriptions[min(layer_index, len(descriptions) - 1)]

    def create_multiple_examples(self, image_paths: List[str], save_dir: str = "figure_1_6_examples"):
        """Create Figure 1.6 visualizations for multiple images."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        print("Creating Figure 1.6 style visualizations...")
        print("="*50)

        for i, img_path in enumerate(image_paths):
            if Path(img_path).exists():
                output_file = save_path / f"figure_1_6_example_{i+1}_{Path(img_path).stem}.png"
                self.create_figure_1_6_visualization(img_path, str(output_file))
                print(f"✓ Created visualization {i+1}: {output_file.name}")
            else:
                print(f"✗ Image not found: {img_path}")

        print(f"\n✓ All Figure 1.6 visualizations saved to: {save_path}")

# Example usage and demo
if __name__ == "__main__":
    # Create Figure 1.6 recreator
    recreator = Figure16Recreator()

    # Find available sample images
    sample_images = [
        "uploads/sample_image.png",
        "uploads/Batch_Directory_1_-_Classifier_Test_Image_1.png",
        "uploads/Single_Image_3_-_Classifier_Test.png"
    ]

    # Filter existing images
    existing_images = [img for img in sample_images if Path(img).exists()]

    if existing_images:
        # Create visualizations for existing images
        recreator.create_multiple_examples(existing_images)
    else:
        print("No sample images found. Creating example with synthetic image...")
        # Create single example with synthetic image
        recreator.create_figure_1_6_visualization(
            "synthetic_waste_image.png",
            "figure_1_6_waste_classification_example.png"
        )
