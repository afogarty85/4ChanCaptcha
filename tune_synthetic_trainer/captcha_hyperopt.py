# redo with torchvision example
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import base64
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
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
from torch.utils.data import Dataset
from PIL import Image
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)




from scipy.special import logsumexp  # log(p1 + p2) = logsumexp([log_p1, log_p2])

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01



def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels

def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def greedy_decode(emission_log_prob, blank=0, **kwargs):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels



def ctc_decode(log_probs, label2char=None, blank=0, method='beam_search', beam_size=10):
    if log_probs.requires_grad == True:
        emission_log_probs = np.transpose(log_probs.detach().cpu().numpy(), (1, 0, 2))
    else:
        emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoders = {
        'greedy': greedy_decode,
        'beam': beam_search_decode,

    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list

# set seed for reproducibility
torch.backends.cudnn.deterministic = True
seed = 823
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

CHARS = '0248ADGHJKMNPRSTVWXY'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}




class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)



def train(model, dataloader, optimizer, criterion, device):

    # tqdm stats
    # pbar_total = len(dataloader)
    # pbar = tqdm(total=pbar_total, desc="Training", position=0, leave=True)

    # set model to train
    model.train()

    # loss info
    tot_count = 0
    tot_loss = 0
    tot_correct = 0

    for data in dataloader:

            # unpack results
        images, targets, target_lengths = [d.to(device) for d in data]

        # forward
        logits = model(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        # aux info
        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

        # loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        tot_count += batch_size
        tot_loss += loss.item()

        # pred
        preds = ctc_decode(log_probs, method='greedy', beam_size=10)
        reals = targets.cpu().numpy().tolist()
        target_lengths = target_lengths.cpu().numpy().tolist()

        target_length_counter = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            if pred == real:
                tot_correct += 1

        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    #     pbar.update(1)
    # pbar.close()

    train_results = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'correct': tot_correct
        }
    return train_results


def evaluate(model, dataloader, optimizer, criterion, device):

    # tqdm stats
    # pbar_total = len(dataloader)
    # pbar = tqdm(total=pbar_total, desc="Evaluating", position=0, leave=True)

    # set model to train
    model.eval()

    # loss info
    tot_count = 0
    tot_loss = 0
    tot_correct = 0

    with torch.no_grad():
        for data in dataloader:

            # unpack results
            images, targets, target_lengths = [d.to(device) for d in data]

            # forward
            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            # aux info
            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            # loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            tot_count += batch_size
            tot_loss += loss.item()

            # pred
            preds = ctc_decode(log_probs, method='greedy', beam_size=10)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1


        #     pbar.update(1)
        # pbar.close()

    eval_results = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'correct': tot_correct
        }
    return eval_results




# load ds
class SynthCaptcha(Dataset):
    CHARS = '0248ADGHJKMNPRSTVWXY'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir, img_height, img_width):

        self.paths = glob.glob(root_dir + '*.png')
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # gray-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        
        image = image.resize((self.img_width, self.img_height), resample=Image.Resampling.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        #image = image / 255.0
        image = torch.FloatTensor(image)

        target = path.split('/')[-1].split('_')[0]
        target_length = path.split('/')[-1].split('_')[1]

        target = [self.CHAR2LABEL[c] for c in list(target)]
        target_length = int(target_length)
        target = torch.LongTensor(target)
        target_length = torch.LongTensor([target_length])
        return image, target, target_length

def synth_collage_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


# get ds
CaptchaSet = SynthCaptcha(root_dir='/mnt/c/Users/afogarty/Desktop/captcha/dataset/fourCaptchas/images/',
                            img_height=80,
                            img_width=300)

# view img
plt.imshow(  CaptchaSet.__getitem__(0)[0].permute(1, 2, 0)  , 'gray' )


# set train, valid, and test size
train_size = int(0.90 * len(CaptchaSet))
valid_size = int(0.10 * len(CaptchaSet)) +1

# use random split to create two data sets;
train_set, val_set = torch.utils.data.random_split(CaptchaSet, [train_size, valid_size])

# init training info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = len(LABEL2CHAR) + 1

# hyperopt;
search_space = {
                'batch_size': hp.choice('batch_size', [32, 64, 96, 128, 256]),
                'lr': hp.uniform('lr', low=0.0001, high=0.0009),
                'seq_to_hidden': hp.quniform('seq_to_hidden', 64, 256, 8),
                'rnn_hidden': hp.quniform('rnn_hidden', 128, 512, 8),
                }


def train_fn(params):
    '''
    HyperOpt 
    '''
    try:

        print(f'Trying these params {params.items()}')

        tot_acc = 0.0
        epochs = 9
        seq_to_hidden = int(params['seq_to_hidden'])
        rnn_hidden = int(params['rnn_hidden'])
        model = CRNN(img_channel=1, img_height=80, img_width=300, num_class=num_classes,
                 map_to_seq_hidden=seq_to_hidden, rnn_hidden=rnn_hidden, leaky_relu=True).to(device)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'])
        criterion = nn.CTCLoss(reduction='sum', zero_infinity=True).to(device)

        # loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8, collate_fn=synth_collage_fn)


        # training loop
        for i, epoch in enumerate(range(1, epochs + 1)):
            train_results = train(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
            tot_acc += train_results['correct']          
            if (epoch == 7) & (tot_acc <= 0):
                print('ending trial, nothing of significance found')
                break
            
        
        print(f'finished with total acc {-tot_acc} and params {params.items()}')

    except Exception as e:
        print(f'Exception: {e}')
    return {'loss': -tot_acc, 'status': STATUS_OK}


# run
argmin = fmin(fn=train_fn,
                space=search_space,
                algo=tpe.suggest,
                max_evals=700,
                early_stop_fn=no_progress_loss(iteration_stop_count=40,
                                                percent_increase=1.0)
                ) 

print('data gen complete')

print(argmin)