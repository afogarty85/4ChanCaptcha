import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import base64
import numpy as np
from tqdm import tqdm
from itertools import groupby
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time, os, copy, shutil
import os
import copy
import cv2
import shutil
import glob
from tqdm import tqdm
import random
import pandas as pd
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import CTCLoss
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# set seed for reproducibility
torch.backends.cudnn.deterministic = True
seed = 823
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    return background


# need to generate spatial dicts for types
char_paths_loc = r"/mnt/c/Users/afogarty/Desktop/captcha/dataset/fourCharacters/images/*/*.png"
char_paths = glob.glob(char_paths_loc)

CHARS = '0248ADGHJKMNPRSTVWXY'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

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


def write_ellipse(image, center_coordinates, axes_length, angle, start_angle, end_angle, color, thickness):
    '''
    Write an Ellipse onto image, mimicking captcha
    '''
    image = cv2.ellipse(image, center_coordinates, axes_length, angle,
                          start_angle, end_angle, color, thickness)
    return image



def generate_random_background(df):
    '''
    /mnt/c/Users/afogarty/Desktop/captcha/captcha_gold/captcha_gold_dataset.csv
    '''
    # get a single row
    sampled = df.sample(n=1)
    # turn to img
    background, foreground = prepare_b64_image(sampled['bg'].values, sampled['fg'].values)
    # join
    out = add_transparent_image(background, foreground, 0, 0)
    # resize
    resized = cv2.resize(out, (300, 80), interpolation=cv2.INTER_NEAREST)
    # gray
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype('uint8')
    # thresh
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 44)
    # Find contours and remove noise
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        # but pick a random selection
        rand_item = random.randint(1, 455)
        if area > rand_item:
            cv2.drawContours(thresh, [c], -1, 0, -1)

    # invert colors again
    invert = cv2.bitwise_not(thresh)

    # stack 3 channels
    out = cv2.merge((invert, invert, invert))

    # reassign to gray
    out[out == 255] = 238

    return out


def load_background_image(background_paths):
    random_path = random.choice(background_paths)
    img = cv2.imread(random_path)
    return img


def generate_synthetic_data(df, config_mapping_type, background_paths, blocker_preds, ellipse=False):
    '''
    Generate synthetic training data
    '''

    # unpack config
    resize_dim = config_mapping[config_mapping_type]['resize_dim']
    x_list = config_mapping[config_mapping_type]['x']
    y_list = config_mapping[config_mapping_type]['y']
    x_jitter = [random.randint(-1, 2) for i in range(len(x_list))]
    y_jitter = [random.randint(-5, 5) for i in range(len(y_list))]
    #n_blockers = random.randint(0, len(x_list))
    n_blockers = blocker_preds[0]

    # load background
    background = generate_random_background(df)
    
    # ensure max size; width (263 is median of sample, 300 is max), height
    background = cv2.resize(background, (300, 80), interpolation=cv2.INTER_NEAREST)

    # label storage
    gold_label = []

    for i, (x, y) in enumerate(zip(x_list, y_list)):

        # load random character
        random_path = random.choice(char_paths)
        img = cv2.imread(random_path)

        # get label
        gold_label_ins = random_path.split('/')[-1][0]

        # map
        gold_label_ins = CHAR2LABEL.get(gold_label_ins)
        assert gold_label_ins != None, 'class mapping failed'

        # append
        gold_label.append(gold_label_ins)

        # resize; (width, height)
        out = cv2.resize(img, resize_dim, interpolation=cv2.INTER_NEAREST)

        # True if zero (black)
        alpha = np.where(out[..., -1] == 0, True, False)

        # set True/False to 0/255 and change type to "uint8" 
        alpha = np.uint8(alpha * 255)

        # stack new alpha layer
        out = np.dstack((out, alpha))

        # add to background
        out = add_transparent_image(background, out, x + x_jitter[i] , y + y_jitter[i])
        
    # write ellipse
    if ellipse:

        for x, y in zip(random.sample(x_list, n_blockers), random.sample(y_list, n_blockers)):

            # generate coords and sizes
            center_coordinates = (x + random.randint(0, 8), y + random.randint(15, 30))  # position, (x, y)
            axes_length = (random.randint(10, 30), random.randint(7, 12))  # height, width

            # write ellipses
            out = write_ellipse(image=out,
                                center_coordinates=center_coordinates,
                                axes_length=axes_length,
                                angle=90,
                                start_angle=0,
                                end_angle=360,
                                color=(150, 150, 150),  # gray
                                thickness=-1)

        # set gray
        #out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    
    # swap ellipse data with other background data
    bg_img = load_background_image(background_paths)

    # find where we are all gray from ellipse
    gray = np.all(out == [150, 150, 150], axis=-1)

    # assign bg vals over the ellipse
    out[gray, :] = bg_img[gray, :]

    # create label len
    gold_label_length = len(gold_label)

    return out, gold_label, gold_label_length


config_mapping = {
    'mid_split': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [20, 50, 85, 170, 210, 255],
    'y': [20, 20, 20, 20, 20, 20]
    },
    'mid_split_left': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [15, 45, 75, 140, 175, 205],
    'y': [20, 20, 20, 20, 20, 20]
    },    
    'one_by_five': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [10, 85, 125, 165, 205, 245],
    'y': [20, 20, 20, 20, 20, 20]
    },
    'five_by_one': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [5, 35, 65, 95, 125, 245],
    'y': [20, 20, 20, 20, 20, 20]
    },
    'two_by_four': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [5, 45, 125, 165, 205, 245],
    'y': [20, 20, 20, 20, 20, 20]
    },
    'four_by_two': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [15, 55, 95, 135, 215, 245],
    'y': [20, 20, 20, 20, 20, 20],
    },
    'one_by_four': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [45, 125, 165, 205, 245],
    'y': [20, 20, 20, 20, 20]
    },
    'four_by_one': {
    'backgrounds': glob.glob("/mnt/c/Users/afogarty/Desktop/captcha/synthetic_generator/gaps/*.png"),
    'resize_dim': (45, 47),  # width, height
    'x': [15, 55, 95, 135, 205],
    'y': [20, 20, 20, 20, 20]
    }
}

# predone captcha bgs with ellipse False
background_paths = glob.glob('/mnt/c/Users/afogarty/Desktop/captcha/generate_synthetic_dataset/backgrounds/*.png')

# load csv
df = pd.read_csv(r'/mnt/c/Users/afogarty/Desktop/captcha/dataset/raw_base64_dataset.csv')

# limit the selection of possible backgrounds
df = df.sample(frac=0.01, replace=False, random_state=88)

# set blocker preds
blocker_preds = np.random.choice([0, 3, 4, 5, 6], 1, p=[0.05, 0.225, 0.225, 0.25, 0.25])

# generate an image, label
out, label, gold_label_length = generate_synthetic_data(df=df,
                                                        config_mapping_type='two_by_four',
                                                        background_paths=background_paths,
                                                        blocker_preds=blocker_preds,
                                                        ellipse=True)
Image.fromarray(out)
print([LABEL2CHAR.get(i) for i in label])

# generate random sequences
folder_loc = '/mnt/c/Users/afogarty/Desktop/captcha/dataset/fourCaptchas/images/'
number_of_sequences = 150000
for i in range(number_of_sequences):
    gen_choice = np.random.choice(list(config_mapping.keys()), 1, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2])
    choice = gen_choice[0]
    # get batch
    out, label, target_length = generate_synthetic_data(
        df=df,
        config_mapping_type=choice,
        background_paths=background_paths,
        blocker_preds=blocker_preds,
        ellipse=False)
    filenamestr = ''.join([LABEL2CHAR.get(x) for x in label]) + '_' + str(target_length) + \
        '_' + str(random.randint(10000, 1000000)) + '.png'
    cv2.imwrite(folder_loc + filenamestr, out)
