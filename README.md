# Autoencoder and Aircraft Damage Detection & Captioning

This repository contains **two separate deep learning projects** focusing on image understanding and generation:

1. **Autoencoder for MNIST Digits** – Demonstrates unsupervised learning and image reconstruction.
2. **Aircraft Damage Detection & Captioning** – Builds a binary classification model using transfer learning (VGG16) and uses BLIP to generate image captions.

-----

##  1: Autoencoder for MNIST Digits

This project explores **autoencoders** for dimensionality reduction and image reconstruction using the **MNIST** handwritten digits dataset.

### Technologies Used

- **Python**
- **TensorFlow (tensorflow.keras)**
- **NumPy**
- **Matplotlib**

### Dataset

- **MNIST**: 60,000 training and 10,000 testing grayscale images of handwritten digits (0–9).

### Workflow

1. **Load and Preprocess Data**
   - Loaded via `tensorflow.keras.datasets.mnist.load_data()`.
   - Normalized to range `[0, 1]` and flattened to 784-dimensional vectors.

2. **Define Autoencoder Architecture**
   - Encoder: Dense layer with ReLU activation.
   - Decoder: Dense layer with Sigmoid activation.
   - Controlled by `encoding_dim`.

3. **Compile and Train Model**
   - Optimizer: `Adam`
   - Loss Function: `Binary Crossentropy`
   - Input images serve as both input and target.

4. **Reconstruct and Visualize**
   - Reconstructed test images shown alongside originals.

-----

##  2: Aircraft Damage Detection & Captioning

This project builds a **binary classifier** for aircraft damage detection and uses the **BLIP** model for generating image captions and summaries.

### Technologies Used

- **Python**
- **TensorFlow (tensorflow.keras)**
- **NumPy**
- **Matplotlib**
- **PIL (Pillow)**
- **PyTorch** (for BLIP)
- **Hugging Face Transformers**
- **ImageDataGenerator** (TensorFlow)
- Utilities: `os`, `tarfile`, `shutil`, `random`, etc.

### Dataset

- **Aircraft Damage Dataset v1**: Contains 'dent' and 'no_dent' subfolders for damaged/undamaged aircraft images.
- Downloaded and extracted programmatically, split into `train`, `valid`, `test`.

### Workflow

1. **Download and Extract Dataset**
   - Downloaded via `urllib`, extracted with `tarfile`.

2. **Setup Directories**
   - Organized into subfolders for training, validation, and testing.

3. **Create Data Generators**
   - Used `ImageDataGenerator` for image resizing, normalization.
   - Set `class_mode='binary'`.

4. **Build Transfer Learning Model (VGG16)**
   - Loaded VGG16 with `include_top=False`.
   - Base layers frozen, custom layers added:
     - Flatten → Dense (ReLU) → Dropout → Dense (Sigmoid)

5. **Compile and Train**
   - Optimizer: Adam
   - Loss: Binary Crossentropy

6. **Evaluate and Visualize**
   - Used `model.evaluate()`.
   - `plot_prediction_image()` visualizes true vs. predicted labels.

7. **Load BLIP Model**
   - Loaded via Hugging Face Transformers.

8. **Generate Captions and Summaries**
   - Function `process_image()`:
     - Opens image
     - Prepares prompt
     - Feeds image to BLIP
     - Returns decoded caption

9. **Test BLIP on Random Images**
   - Randomly selected 10 test images for captioning.

10. **Combine CNN + BLIP Output**
    - Ran classification with CNN.
    - Saved image to temp file for BLIP input.
    - Generated and displayed prediction + caption.

11. **Set Seeds for Reproducibility**
    - Ensured consistency in output and training.

-----

## Learnings from This Notebook

### From the Autoencoder Project

- **Understanding Autoencoders**: Learned basic architecture for encoding and decoding data.
- **Dimensionality Reduction**: Compressed high-dimensional data into a lower-dimensional latent space.
- **Image Reconstruction**: Observed quality of output from compressed inputs.
- **Appropriate Loss Functions**: Used `binary_crossentropy` for normalized pixel reconstruction.

### From the Aircraft Damage Detection & Captioning Project

- **Transfer Learning**: Used VGG16 pre-trained on ImageNet to classify custom aircraft dataset.
- **Fine-Tuning Techniques**: Froze convolutional base, trained custom top layers.
- **ImageDataGenerator**: Learned to load and preprocess image data efficiently.
- **Binary Classification**: Applied CNN to real-world two-class problem.
- **Model Evaluation**: Evaluated using metrics and plotted predictions.
- **BLIP Vision-Language Model**: Explored cutting-edge image captioning with Hugging Face.
- **Captioning and Summarization**: Generated text that explains image context.
- **Model Integration**: Combined CNN classifier with BLIP captioner for enriched output.
- **Real-World Data Handling**: Programmatically downloaded, extracted, and structured dataset.

-----
