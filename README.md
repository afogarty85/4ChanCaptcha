# 4ChanCaptchaDataSet


This repository contains a 4chan captcha data set amounting to 7181 foreground and background images,
along with procedures needed to produce or extend the captcha data set on your own.

Extensions of this project are forthcoming, which include:

1. Hyperparameter turning of opencv2 processing methods to remove noise and extract characters with accuracy
2. Pytesseract training procedures to read 4chan captcha
3. An ML model that predicts the right answer


## Sample Background and Foreground Image

<img title="Background" alt="Background" src="/sample_images/background.png">

<img title="Foreground" alt="Foreground" src="/sample_images/foreground.png">


## Some Starting Code

```
import cv2
import base64
import numpy as np
import pandas as pd


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
    background = cv2.imdecode(background, cv2.IMREAD_COLOR)

    # fg
    foreground = base64.b64decode(str(foreground))
    foreground = np.frombuffer(foreground, np.uint8)
    foreground = cv2.imdecode(foreground, cv2.IMREAD_COLOR)

    # we need its alpha channel
    foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2RGBA)
    return background, foreground


# get data
df = pd.read_csv(r'data\captcha_training_set.csv')

# get the first image
bg_ = df.iloc[0]['bg']
fg_ = df.iloc[0]['fg']

# transform
background, foreground = prepare_b64_image(bg_, fg_)

```
