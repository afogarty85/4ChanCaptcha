# 4ChanCaptcha


This repository contains a 4chan captcha data set amounting to 16242 foreground and background images and the [procedures](/base64_captcha_scraper/base64_generator.py) needed to produce or extend the captcha data set on your own.


<img title="Background" alt="Background" src="/sample_images/background.png">

<img title="Foreground" alt="Foreground" src="/sample_images/foreground.png">


## Included Data

1. [base64 dataset](dataset/raw_base64_dataset.csv) -- the raw data amounting to 16242 foreground and background images

2. [fourCharacters](dataset/fourCharacters/) -- a set of extracted characters, derived from `Character Extractor` below.


It also contains the following modules that aim to exploit the data:

## Character Extractor

This module is the main workhorse which populates the foundation necessary to generate synthetic training data. Using a set of hyperparameters derived from [optimize_character_extractor](character_extractor/optimize_character_extractor.py), [character_extractor.py](character_extractor/character_extractor.py) isolates and extracts individual characters and digits from the image, the atomic unit of our synthetic training data generator.

## Classify Character

[classify_character.py](classify_character/classify_character.py) is a straight forward TorchVision implementation of ResNet18 that classifies extracted characters.

## Generate Synthetic Data Set

[generate_synthetic_dataset.py](generate_synthetic_dataset/generate_synthetic_dataset.py) aims to artificially generate 4chan captchas for the main image model to train on, for example:

<img title="Synthetic Captcha" alt="Synthetic Captcha" src="/sample_images/synthetic.png">


## Train Synthetic Data Set

[train_synthetic_dataset.py](train_synthetic_dataset/train_synthetic_dataset.py) uses a [CRNN](https://arxiv.org/pdf/1507.05717.pdf) model to train on the synthetically generated data set. It also includes a small prediction function as well as a lightly modified model to generate an ONNX export model capable of running in the browser.


## CRNN Browser Powered Inference

The ultimate consequence of this repository is a capable model that is agile enough to be run from your local browser. Using `Tampermonkey` or `Requestly`, you can load the model ([onnx_model.onnx](browser_inference/onnx_model.onnx)) into your own browser using [browser_inference](browser_inference/browser_inference.js).

<img title="Browser Inference" alt="Browser Inference" src="/sample_images/browser_out.png">


### Some Starting Code

```
import cv2
import base64
import numpy as np
import pandas as pd
from PIL import Image


# b64 processing
def prepare_b64_image(background, foreground):
    '''
    params:
    background: string  - base64 image as text
    foreground: string  - base64 image as text
    '''
    # bg
    background = base64.b64decode(str(background))
    background = np.frombuffer(background, np.uint8)
    background = cv2.imdecode(background, cv2.IMREAD_UNCHANGED)

    # fg
    foreground = base64.b64decode(str(foreground))
    foreground = np.frombuffer(foreground, np.uint8)
    foreground = cv2.imdecode(foreground, cv2.IMREAD_UNCHANGED)
    return background, foreground


# get data
df = pd.read_csv(r'data\captcha_training_set.csv')

# get the first image
bg_ = df.iloc[0]['bg']
fg_ = df.iloc[0]['fg']

# transform
background, foreground = prepare_b64_image(bg_, fg_)

# show image
Image.fromarray(np.uint8(background))
Image.fromarray(np.uint8(foreground))
```
