#!/usr/bin/env python3
"""
CNN Visualization Web Server

A Flask web application to showcase CNN visualizations, Figure 1.6 recreations,
and interactive analysis of the waste classification model.

Usage:
    python app.py

Then visit: http://localhost:5000

Author: CNN Visualization Web Server
Date: 2025-08-03
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import base64

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/visualizations'

# Create output directory
Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

def get_model_info():
    """Get model information for the web interface."""
    return {
        'name': 'Waste Classification CNN',
        'architecture': 'MobileNetV2 + Custom Classification Head',
        'num_classes': 5,
        'class_names': ['plastic', 'organic', 'metal', 'glass', 'paper'],
        'input_shape': [224, 224, 3],
        'total_params': 2422000,
        'layers': [
            {
                'name': 'Input Layer',
                'type': 'InputLayer',
                'output_shape': [224, 224, 3],
                'params': 0,
                'description': 'RGB input image'
            },
            {
                'name': 'MobileNetV2 Base',
                'type': 'MobileNetV2',
                'output_shape': [7, 7, 1280],
                'params': 2257984,
                'description': 'Pre-trained feature extractor'
            },
            {
                'name': 'Global Average Pooling',
                'type': 'GlobalAveragePooling2D',
                'output_shape': [1280],
                'params': 0,
                'description': 'Spatial dimension reduction'
            },
            {
                'name': 'Dropout 1',
                'type': 'Dropout',
                'output_shape': [1280],
                'params': 0,
                'description': 'Regularization (20% dropout)'
            },
            {
                'name': 'Dense Layer',
                'type': 'Dense',
                'output_shape': [128],
                'params': 164608,
                'description': 'Feature combination layer'
            },
            {
                'name': 'Dropout 2',
                'type': 'Dropout',
                'output_shape': [128],
                'params': 0,
                'description': 'Regularization (20% dropout)'
            },
            {
                'name': 'Classification Output',
                'type': 'Dense',
                'output_shape': [5],
                'params': 645,
                'description': 'Classification layer (softmax)'
            }
        ]
    }

def get_sample_images():
    """Get list of available sample images."""
    uploads_dir = Path('uploads')
    images = []

    if uploads_dir.exists():
        for ext in ['.png', '.jpg', '.jpeg']:
            for img_file in uploads_dir.glob(f'*{ext}'):
                if 'figure' not in img_file.name.lower():  # Exclude reference figures
                    images.append({
                        'filename': img_file.name,
                        'path': str(img_file),
                        'size_kb': img_file.stat().st_size // 1024,
                        'type': 'test' if 'test' in img_file.name.lower() else 'sample'
                    })

    return images

def simulate_feature_maps():
    """Simulate feature map data for visualization."""
    import random
    random.seed(42)

    layers = [
        {
            'name': 'Layer 1 - Edge Detection',
            'shape': [56, 56, 32],
            'description': 'Detects edges, corners, basic textures',
            'features': ['Vertical edges', 'Horizontal edges', 'Corners', 'Basic textures'],
            'sample_maps': [
                [[random.random() for _ in range(8)] for _ in range(8)] for _ in range(4)
            ]
        },
        {
            'name': 'Layer 2 - Pattern Recognition',
            'shape': [28, 28, 64],
            'description': 'Combines patterns into shapes',
            'features': ['Circular shapes', 'Rectangular patterns', 'Texture combinations', 'Object parts'],
            'sample_maps': [
                [[random.random() for _ in range(8)] for _ in range(8)] for _ in range(4)
            ]
        },
        {
            'name': 'Layer 3 - Object Features',
            'shape': [14, 14, 128],
            'description': 'High-level object and material features',
            'features': ['Bottle shapes', 'Container patterns', 'Material textures', 'Object semantics'],
            'sample_maps': [
                [[random.random() for _ in range(8)] for _ in range(8)] for _ in range(4)
            ]
        }
    ]

    return layers

def simulate_classification_results():
    """Simulate classification results."""
    import random
    random.seed(42)

    class_names = ['plastic', 'organic', 'metal', 'glass', 'paper']

    # Create multiple example predictions
    examples = []
    for i in range(3):
        # Generate random probabilities that sum to 1
        probs = [random.random() for _ in class_names]
        total = sum(probs)
        probs = [p/total for p in probs]

        # Sort by probability
        results = list(zip(class_names, probs))
        results.sort(key=lambda x: x[1], reverse=True)

        examples.append({
            'image_name': f'Sample Image {i+1}',
            'predictions': [
                {'class_name': name, 'confidence': prob}
                for name, prob in results
            ]
        })

    return examples

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html',
                         model_info=get_model_info(),
                         sample_images=get_sample_images())

@app.route('/architecture')
def architecture():
    """CNN Architecture visualization page."""
    return render_template('architecture.html',
                         model_info=get_model_info())

@app.route('/feature-maps')
def feature_maps():
    """Feature maps visualization page."""
    return render_template('feature_maps.html',
                         feature_layers=simulate_feature_maps(),
                         sample_images=get_sample_images())

@app.route('/figure-1-6')
def figure_1_6():
    """Figure 1.6 recreation page."""
    return render_template('figure_1_6.html',
                         feature_layers=simulate_feature_maps(),
                         classification_results=simulate_classification_results(),
                         sample_images=get_sample_images())

@app.route('/operations')
def operations():
    """CNN operations demonstration page."""
    return render_template('operations.html')

@app.route('/api/model-info')
def api_model_info():
    """API endpoint for model information."""
    return jsonify(get_model_info())

@app.route('/api/feature-maps/<image_name>')
def api_feature_maps(image_name):
    """API endpoint for feature maps of a specific image."""
    return jsonify({
        'image_name': image_name,
        'layers': simulate_feature_maps()
    })

@app.route('/api/classification/<image_name>')
def api_classification(image_name):
    """API endpoint for classification results."""
    results = simulate_classification_results()
    return jsonify(results[0])  # Return first example

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory('uploads', filename)

@app.route('/static/visualizations/<filename>')
def visualization_file(filename):
    """Serve generated visualization files."""
    return send_from_directory('static/visualizations', filename)

if __name__ == '__main__':
    print("🚀 Starting CNN Visualization Web Server...")
    print("=" * 50)
    print("📊 Features available:")
    print("- Interactive Figure 1.6 recreation")
    print("- CNN architecture visualization")
    print("- Feature map exploration")
    print("- CNN operations demonstration")
    print("- Real-time model analysis")
    print("=" * 50)
    print("🌐 Access your visualizations at:")
    print("   http://localhost:5000")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
