# Image Classification with ResNet

This project implements a custom ResNet model for image classification using PyTorch, with a Streamlit interface for easy interaction. The model is trained on the CIFAR-10 dataset and includes model explainability using GradCAM.

## Features

- Custom ResNet implementation with PyTorch
- CIFAR-10 dataset training with 10 classes:
  - airplane, automobile, bird, cat, deer
  - dog, frog, horse, ship, truck
- Interactive Streamlit interface with:
  - File upload capability
  - Real-time camera capture
  - Visual confidence scores
- Performance optimizations:
  - Parallel data loading
  - GPU acceleration (when available)
  - Batched processing
  - Learning rate scheduling
- Model explainability with GradCAM

## Live Demo

Try the live demo at: [Streamlit Cloud App](https://image-classification-resnet.streamlit.app)

## Project Structure

- `app.py`: Streamlit application for inference
- `resnet_image_classification.ipynb`: Training notebook with model implementation
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/ShaLese/Hugging-Face-Projects.git
cd image_classification_with_ResNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser
2. Choose input method:
   - Upload an image file (JPG, JPEG, PNG)
   - Take a photo using your camera
3. View predictions with confidence scores
4. Results show top 3 predictions with probability bars

## Model Architecture

The custom ResNet implementation includes:
- Multiple residual blocks
- Batch normalization
- Adaptive average pooling
- Device-agnostic code (runs on CPU or GPU)

## Performance Features

- Parallel data loading with multiple workers
- GPU acceleration when available
- Batched processing for efficient computation
- Learning rate scheduling with cosine annealing
- Progress tracking with tqdm
- Execution time monitoring

## Model Explainability

The project includes GradCAM implementation for visualizing model decisions, helping understand which parts of the image influenced the classification.

## Deployment

The app is deployed on Streamlit Cloud for easy access. To deploy your own version:

1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy directly from the repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.
