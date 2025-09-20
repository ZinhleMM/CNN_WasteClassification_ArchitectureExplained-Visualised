#!/usr/bin/env python3
"""
CNN Visualization Demo Script

This script demonstrates all CNN visualization tools created for the waste
classification model. It provides a comprehensive showcase of CNN architecture,
layer transformations, feature maps, and data flow visualizations.

Run this script to generate all visualizations inspired by Figure 1.6 and
the resources from GeeksforGeeks, DataCamp, and Keras documentation.

Usage:
    python cnn_visualization_demo.py

References:
- Figure 1.6: Data representations learned by a digit-classification model
- GeeksforGeeks CNN Introduction
- DataCamp CNN Tutorial
- Deep Learning with Python by François Chollet

Author: CNN Visualization Demo
Date: 2025-08-03
"""

import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import our visualization modules
from cnn_architecture_visualizer import CNNArchitectureVisualizer
from feature_map_extractor import FeatureMapExtractor
from interactive_cnn_analyzer import InteractiveCNNAnalyzer
from figure_1_6_recreator import Figure16Recreator

def print_banner():
    """Print welcome banner."""
    print("="*80)
    print("CNN VISUALIZATION TOOLKIT FOR WASTE CLASSIFICATION")
    print("="*80)
    print("Comprehensive visualization of CNN architecture, layers, and transformations")
    print("Inspired by Figure 1.6 and CNN tutorials from GeeksforGeeks & DataCamp")
    print("="*80)
    print()

def find_available_files():
    """Find available model and image files."""
    files = {
        'model_path': None,
        'class_names_path': None,
        'sample_images': [],
        'all_images': []
    }

    # Look for model files
    model_candidates = [
        'uploads/waste_model.keras',
        'waste_classifier/model/waste_model.keras',
        'model/waste_model.keras'
    ]

    for path in model_candidates:
        if Path(path).exists():
            files['model_path'] = path
            break

    # Look for class names
    classnames_candidates = [
        'uploads/classnames.json',
        'waste_classifier/model/classnames.json',
        'model/classnames.json'
    ]

    for path in classnames_candidates:
        if Path(path).exists():
            files['class_names_path'] = path
            break

    # Find sample images
    image_extensions = ['.png', '.jpg', '.jpeg']
    uploads_dir = Path('uploads')

    if uploads_dir.exists():
        for ext in image_extensions:
            files['all_images'].extend(list(uploads_dir.glob(f'*{ext}')))
            files['all_images'].extend(list(uploads_dir.glob(f'*{ext.upper()}')))

    # Select representative sample images
    priority_names = ['sample_image', 'test', 'classifier', 'batch']
    for priority in priority_names:
        matching_images = [img for img in files['all_images']
                         if priority.lower() in img.name.lower()]
        if matching_images:
            files['sample_images'].extend(matching_images[:2])  # Max 2 per category

    # Ensure we have at least some sample images
    if not files['sample_images'] and files['all_images']:
        files['sample_images'] = files['all_images'][:3]  # Take first 3

    return files

def load_class_names(class_names_path):
    """Load class names from JSON file."""
    try:
        with open(class_names_path, 'r') as f:
            return json.load(f)
    except:
        return ['plastic', 'organic', 'metal', 'glass', 'paper']

def create_model_info(model_path, class_names):
    """Create model information dictionary."""
    return {
        'model_path': model_path,
        'num_classes': len(class_names),
        'class_names': class_names,
        'input_shape': (224, 224, 3),
        'architecture': 'MobileNetV2 + Custom Classification Head',
        'transfer_learning': True
    }

def demo_architecture_visualization(model_info):
    """Demonstrate CNN architecture visualization."""
    print("1. CNN ARCHITECTURE VISUALIZATION")
    print("-" * 40)

    visualizer = CNNArchitectureVisualizer()

    print("Creating comprehensive CNN architecture visualizations...")

    # Create all architecture visualizations
    visualizer.create_complete_cnn_visualization(
        model_info,
        save_dir="demo_output/01_architecture"
    )

    print("✓ Architecture visualizations completed!")
    print()

def demo_feature_extraction(model_path, sample_images):
    """Demonstrate feature map extraction and visualization."""
    print("2. FEATURE MAP EXTRACTION & VISUALIZATION")
    print("-" * 45)

    extractor = FeatureMapExtractor(model_path)

    # Process sample images
    output_dir = Path("demo_output/02_feature_maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(sample_images[:2]):  # Limit to 2 images
        print(f"Processing image {i+1}: {img_path.name}")

        # Extract feature maps
        feature_data = extractor.extract_feature_maps(str(img_path))

        # Create visualizations
        extractor.visualize_feature_maps(
            feature_data,
            save_path=str(output_dir / f"feature_maps_{i+1}_{img_path.stem}.png")
        )

        extractor.visualize_layer_progression(
            feature_data,
            save_path=str(output_dir / f"layer_progression_{i+1}_{img_path.stem}.png")
        )

    # Compare multiple images if available
    if len(sample_images) > 1:
        print("Creating image comparison...")
        extractor.compare_multiple_images(
            [str(img) for img in sample_images[:3]],
            save_path=str(output_dir / "feature_comparison.png")
        )

    print("✓ Feature map visualizations completed!")
    print()

def demo_interactive_analysis(model_path, class_names, sample_image):
    """Demonstrate interactive CNN analysis."""
    print("3. INTERACTIVE CNN ANALYSIS")
    print("-" * 30)

    analyzer = InteractiveCNNAnalyzer(model_path, class_names)

    print("Creating comprehensive CNN analysis...")

    # Create complete analysis
    sample_image_path = str(sample_image) if sample_image else None
    analyzer.create_comprehensive_analysis(
        sample_image_path=sample_image_path,
        save_dir="demo_output/03_analysis"
    )

    print("✓ Interactive analysis completed!")
    print()

def demo_figure_1_6_recreation(model_path, sample_images):
    """Demonstrate Figure 1.6 style recreation."""
    print("4. FIGURE 1.6 STYLE RECREATION")
    print("-" * 32)

    recreator = Figure16Recreator(model_path)

    print("Creating Figure 1.6 style visualizations...")

    # Create visualizations for sample images
    if sample_images:
        recreator.create_multiple_examples(
            [str(img) for img in sample_images[:3]],
            save_dir="demo_output/04_figure_1_6"
        )
    else:
        # Create with synthetic image
        recreator.create_figure_1_6_visualization(
            "synthetic_waste_image.png",
            "demo_output/04_figure_1_6/figure_1_6_synthetic.png"
        )

    print("✓ Figure 1.6 recreations completed!")
    print()

def create_summary_report(files, model_info):
    """Create a summary report of all generated visualizations."""
    print("5. GENERATING SUMMARY REPORT")
    print("-" * 30)

    output_dir = Path("demo_output")
    summary_file = output_dir / "00_visualization_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("CNN VISUALIZATION TOOLKIT - SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("Generated for Waste Classification CNN Model\n")
        f.write(f"Inspired by Figure 1.6 and CNN tutorials\n\n")

        f.write("MODEL INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model Path: {files['model_path'] or 'Simulated'}\n")
        f.write(f"Number of Classes: {model_info['num_classes']}\n")
        f.write(f"Class Names: {', '.join(model_info['class_names'])}\n")
        f.write(f"Input Shape: {model_info['input_shape']}\n")
        f.write(f"Architecture: {model_info['architecture']}\n\n")

        f.write("SAMPLE IMAGES PROCESSED:\n")
        f.write("-" * 25 + "\n")
        for img in files['sample_images']:
            f.write(f"- {img.name}\n")
        f.write("\n")

        f.write("GENERATED VISUALIZATIONS:\n")
        f.write("-" * 25 + "\n")

        # List all generated files
        for subdir in sorted(output_dir.glob("*")):
            if subdir.is_dir():
                f.write(f"\n{subdir.name}:\n")
                for file in sorted(subdir.glob("*")):
                    f.write(f"  - {file.name}\n")

        f.write("\nVISUALIZATION DESCRIPTIONS:\n")
        f.write("-" * 28 + "\n")
        f.write("01_architecture: Complete CNN architecture diagrams and layer analysis\n")
        f.write("02_feature_maps: Feature map extractions and layer transformations\n")
        f.write("03_analysis: Interactive CNN analysis with detailed metrics\n")
        f.write("04_figure_1_6: Figure 1.6 style recreations showing data flow\n")

        f.write("\nREFERENCES:\n")
        f.write("-" * 11 + "\n")
        f.write("- Figure 1.6: Data representations learned by a digit-classification model\n")
        f.write("- GeeksforGeeks: Introduction to Convolution Neural Network\n")
        f.write("- DataCamp: Convolutional Neural Networks in Python\n")
        f.write("- Deep Learning with Python by François Chollet\n")

    print(f"✓ Summary report created: {summary_file}")
    print()

def main():
    """Main demo function."""
    print_banner()

    # Find available files
    print("Scanning for available files...")
    files = find_available_files()

    # Load class names
    class_names = ['plastic', 'organic', 'metal', 'glass', 'paper']  # Default
    if files['class_names_path']:
        class_names = load_class_names(files['class_names_path'])

    # Create model info
    model_info = create_model_info(files['model_path'], class_names)

    # Print file status
    print(f"Model file: {'✓' if files['model_path'] else '✗'} {files['model_path'] or 'Not found (will simulate)'}")
    print(f"Class names: {'✓' if files['class_names_path'] else '✗'} {files['class_names_path'] or 'Using defaults'}")
    print(f"Sample images: {len(files['sample_images'])} found")
    if files['sample_images']:
        for img in files['sample_images'][:3]:
            print(f"  - {img.name}")
    print()

    # Create output directory
    Path("demo_output").mkdir(exist_ok=True)

    # Run demonstrations
    try:
        # 1. Architecture visualization
        demo_architecture_visualization(model_info)

        # 2. Feature map extraction
        demo_feature_extraction(files['model_path'], files['sample_images'])

        # 3. Interactive analysis
        sample_image = files['sample_images'][0] if files['sample_images'] else None
        demo_interactive_analysis(files['model_path'], class_names, sample_image)

        # 4. Figure 1.6 recreation
        demo_figure_1_6_recreation(files['model_path'], files['sample_images'])

        # 5. Summary report
        create_summary_report(files, model_info)

        # Final message
        print("🎉 CNN VISUALIZATION DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("All visualizations have been created in the 'demo_output' directory.")
        print("Check the summary report for detailed information about generated files.")
        print()
        print("Generated visualization categories:")
        print("- CNN Architecture diagrams and layer analysis")
        print("- Feature map extractions and transformations")
        print("- Interactive CNN analysis with detailed metrics")
        print("- Figure 1.6 style recreations showing data representations")
        print()
        print("These visualizations demonstrate how your waste classification CNN")
        print("processes images through successive layers, extracting increasingly")
        print("complex features leading to final classification.")

    except Exception as e:
        print(f"❌ Error during demo execution: {e}")
        print("Please check the error details and try again.")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
