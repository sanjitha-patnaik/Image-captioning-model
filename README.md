# Image-captioning-model
An image captioning project that uses deep learning with CNNs and LSTMs to generate descriptive captions for images

# Image Captioning with CNN and LSTM

This project implements an image captioning model using Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory (LSTM) networks for generating captions. The model leverages a pre-trained ResNet50 for extracting image features and a custom LSTM-based architecture to generate descriptive captions from those features. The dataset used in this project is the Microsoft COCO dataset, which contains images paired with human-generated captions.

## Project Overview

In this project:
- **ResNet50** is used to extract image features by removing the final classification layers.
- **LSTM** is employed to generate captions based on the image features and the sequence of words.
- **BERT Tokenizer** is used to tokenize captions and prepare them for training.
- The model is trained using cross-entropy loss and evaluated using the BLEU score metric to measure the quality of the generated captions.

## Key Features
- **Image Feature Extraction**: Utilizes a pre-trained ResNet50 model to extract high-level image features.
- **Caption Generation**: Uses an LSTM network to generate captions based on image features.
- **Data Processing**: Processes the COCO dataset and tokenizes captions using a BERT tokenizer.
- **Evaluation**: BLEU score is calculated to evaluate the performance of the captioning model on the validation dataset.

## Requirements
- Python 3.12
- PyTorch
- Transformers
- NLTK
- Pillow
- tqdm
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-captioning.git
