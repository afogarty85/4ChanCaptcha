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



# prepare data function
def prepare_data(train_ratio, root_dir, project_name):
    '''
    This function sorting and creating of a training and test set


    Parameters
    ------------
    train_ratio : float
                The proportion of images that should be used for training.
                Validation data is taken from the training set.

    root_dir : string
                Location to the root folder of where the images are, e.g.,:
                'C:\\Users\\Andrew\\.data'

    project_name : string
                The name of the project folder inside the root folder which has
                an 'images' folder that contains sub-folders for each class,
                e.g., a cat folder, a dog folder, etc.

    Returns
    ------------
    None

    '''

    # establish pathing
    data_dir = os.path.join(root_dir, project_name)
    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # make directories and find out the number of classes
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    classes = os.listdir(images_dir)

    # for each class, search over the files and copy them to new dirs
    for c in classes:
        class_dir = os.path.join(images_dir, c)
        images = os.listdir(class_dir)
        n_train = int(len(images) * train_ratio)
        train_images = images[:n_train]
        test_images = images[n_train:]
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)

        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image)
            shutil.copyfile(image_src, image_dst)

        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image)
            shutil.copyfile(image_src, image_dst)

    return

# prepare data
root_dir = '/mnt/c/Users/afogarty/Desktop/captcha/dataset'
prepare_data(train_ratio=0.8,
             root_dir=root_dir,
             project_name='fourCharacters')



def train_model(model, dataloaders, criterion, optimizer, num_epochs, crops):
    '''
    This function handles training, validation, and the recording of training
    data


    Parameters
    ------------
    model : object
                A PyTorch model based on the torch.nn.Module.

    dataloaders : object
                A PyTorch data loader.

    criterion : object
                The loss function, generally cross_entropy.

    optimizer : object
                A compatible Torch optimizer.

    num_epochs : int
                The number of training epochs.

    Returns
    ------------
    model : object
            A PyTorch model
    input_size  : int
            The architecture's image requirement; useful for transformations
    '''
    since = time.time()

    # save metrics
    val_acc_history = []
    s_labels, s_preds, s_inputs = [], [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # collect stats
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        if crops:
                            # for n-crops
                            bs, ncrops, c, h, w = inputs.size()
                            # forward
                            outputs = model(inputs.view(-1, c, h, w))  # fuse batch size and ncrops
                            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # average the output over ncrops
                            loss = criterion(outputs_avg, labels)
                            # preds
                            _, preds = torch.max(outputs_avg, 1)
                        else:
                            # forward
                            outputs = model(inputs)
                            loss = F.cross_entropy(outputs, torch.flatten(labels))
                            # preds
                            _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        # forward
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # get preds
                        _, preds = torch.max(outputs, 1)
                        # store metrics
                        s_labels.extend(labels.cpu())
                        s_preds.extend(preds.cpu())
                        s_inputs.extend(inputs.cpu())

                # batch statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print('saving new weights...')
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print('')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights once done training
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, s_labels, s_preds, s_inputs


def initialize_model(model_name, num_classes, weights, grayscale=True):
    '''
    This function loads model architecture and establishes its use
    so that it matches the right transformations later on.


    Parameters
    ------------
    model_name : string
                A supported TorchVision model, e.g.,:
                'resnet', 'inception', 'lenet', etc.

    num_classes : int
                The number of classes being predicted.

    weights : str
                ResNet18_Weights.DEFAULT for best available from ImageNet.

    grayscale : bool
                Whether or not we have grayscale images.

    Returns
    ------------
    model : object
            A PyTorch model

    '''
    model = None

    if model_name == "resnet":
        model = models.resnet18(weights=weights)
        # reassign grayscale
        if grayscale: # (1, 64 vs 3, 64)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # reassign num classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_normalization(train_folder):
    '''
    This function finds the values to normalize
    our training images in the absence of a pre-trained model.
    If we use a pre-trained model, we must use their normalizing values.


    Parameters
    ------------
    train_folder : string
                A string file path to the base train folder,
                e.g.,: .data\\hymenoptera_data\\train

    Returns
    ------------
    means : tensor
            Channel means
    stds  : tensor
            Channel stds
    '''
    # loads data in shape: (channels, height, width)
    train_data = datasets.ImageFolder(root=train_folder,
                                      transform=transforms.ToTensor())

    # storage containers
    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, _ in train_data:
        # sum the means and stds; then find average
        means += torch.mean(img, dim=(1,2))
        stds += torch.std(img, dim=(1,2))

    means /= len(train_data)
    stds /= len(train_data)

    return means, stds


#run normalization if not pre-trained model
# root_dir = "captcha/dataset/fourCharacters"
# means, stds = get_normalization(root_dir)
# mean; ([0.4787, 0.4787, 0.4787])
# std; ([0.4520, 0.4520, 0.4520])


# set input size; resnet is established with 224, 224
input_size = (224, 224)

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        torchvision.transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(input_size[0], scale=(0.6, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation((-15, 15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4787], std=[0.4520])  # in case we want normalization on our own terms
        # transforms.TenCrop(input_size),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda tensors:
        #                   torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.4293, 0.4293, 0.4293])(t) for t in tensors]))

    ]),
    # limited transforms for validation
    'test': transforms.Compose([
        transforms.Resize(input_size),
        torchvision.transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4787], std=[0.4520])
    ]),
}


# prepare weighted sampling for imbalanced classification
def create_sampler(train_ds):
    # extract labels
    labels = [batch[1] for batch in train_ds]
    # generate class distributions [y1, y2, etc...]
    bin_count = np.bincount(labels)
    # weight gen
    weight = 1. / bin_count.astype(np.float32)
    # produce weights for each observation in the data set
    samples_weight = torch.tensor([weight[t] for t in labels])
    # prepare sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                        num_samples=len(samples_weight),
                                                        replacement=True)
    return sampler



def build_datasets_loaders(root_dir, data_transforms, valid_ratio, batch_size, num_workers):
    '''
    This function builds torch image data sets and data loaders


    Parameters
    ------------
    root_dir : string
                A string file path to the root directory,
                e.g.,: .data\\hymenoptera_data

    data_transforms : dict
                    torchvision.transforms operations to perform

    valid_ratio : float
                    A decimal percentage indicating the amount of validation data

    batch_size : int
                    The size of the data loader batching

    num_workers : int
                    Number of CPU cores to load data with

    Returns
    ------------
    img_datasets : dict
            A dictionary of training and validation data sets

    dataloaders_dict  : dict
            A dictionary of data loaders for each data set
    '''
    # create training and validation datasets;
    img_sets = {
        x: (
            datasets.ImageFolder(root=os.path.join(root_dir, x),
                                #  apply transforms
                                transform=data_transforms[x])
           )
        for x in ['train', 'test']
        }
    # setup conditions for validation data
    n_train_examples = int(len(img_sets['train']) * valid_ratio)
    n_valid_examples = len(img_sets['train']) - n_train_examples
    # random split
    train_data, valid_data = data.random_split(img_sets['train'],
                                               [n_train_examples, n_valid_examples])
    # make a deepcopy to not worry about wrong transforms
    valid_data = copy.deepcopy(valid_data)
    # apply test transforms
    valid_data.dataset.transform = data_transforms['test']
    # apply train transforms
    train_data.dataset.transform = data_transforms['train']

    # repackage
    img_sets = {
        x: y
        for x, y in zip(['train', 'val', 'test'],
                        (train_data, valid_data, img_sets['test']))
        }

    # create weighted sampler
    train_sampler = create_sampler(img_sets['train'])

    # create data loaders
    dataloaders_dict = {
        x: (torch.utils.data.DataLoader(img_sets[x],
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True,  # dont keep imbalanced batch
                                        num_workers=num_workers)
            )
        for x in ['val', 'test']
        }

    # separate train instance so we can add a sampler for imbalanced classes
    dataloaders_dict['train'] = torch.utils.data.DataLoader(img_sets['train'],
                                        batch_size=batch_size,
                                        sampler=train_sampler,
                                        drop_last=True,  # dont keep imbalanced batch
                                        num_workers=num_workers)

    return img_sets, dataloaders_dict


# create data sets and data loaders
root_dir = "/mnt/c/Users/afogarty/Desktop/captcha/dataset/fourCharacters"
img_datasets, dataloaders = build_datasets_loaders(root_dir,
                                                   data_transforms=data_transforms,
                                                   valid_ratio=0.8,
                                                   batch_size=32,
                                                   num_workers=4)


# get a batch
dataiter = iter(dataloaders['train'])
batch = next(dataiter)

# print shapes
print(f'Image shape: {batch[0].shape}')
print(f'Label shape: {batch[1].shape}')



# initialize the model
model = initialize_model(model_name='resnet',
                                     num_classes=20,
                                     weights=None,
                                     grayscale=True)


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# send the model to device
model = model.to(device)
# create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# create loss function
criterion = nn.CrossEntropyLoss()
# train and evaluate
model, val_acc_history, labels, preds, images = train_model(
    model, dataloaders, criterion, optimizer, num_epochs=50, crops=False)


# get preds
def get_predictions(model, iterator):
    model.eval()
    images = []
    labels = []
    probs = []
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    return images, labels, probs


# get images, labels, and probabilities
images, labels, probs = get_predictions(model, dataloaders['test'])
# get predicted labels
pred_labels = torch.argmax(probs, 1)
# correct images
corrects = torch.eq(labels, pred_labels)
# incorrect ones - storage
incorrect_examples = []
# correct examples - storage
correct_examples = []
# append incorrect, label, and its probability
for image, label, prob, correct in zip(images, labels, probs, corrects):
    if not correct:
        incorrect_examples.append((image.permute(1, 2, 0).squeeze() * 255, label, prob))
    else:
        correct_examples.append((image.permute(1, 2, 0).squeeze() * 255, label, prob))

# total acc
print('Test Acc:', corrects.sum() / len(labels) )

incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
correct_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

# view an image
Image.fromarray(np.uint8(incorrect_examples[0][0]))


def plot_predictions(examples, classes, n_images):
    '''
    This function plots predictions

    Parameters
    ------------
    examples : torch tensor
                A tensor containing all test examples

    classes : list
                A list containing the classes from the image data set

    n_images : int
                Number of images to display; rounded by int(np.sqrt(n_images))

    Returns
    ------------
    None
    '''
    # establish figure
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize = (25, 20))
    for i in range(rows*cols):
        # gen subplot
        ax = fig.add_subplot(rows, cols, i+1)
        # extract results
        image, true_label, probs = examples[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim = 0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]
        # transform to image
        ax.imshow(image.cpu().numpy(), cmap='gray')
        # set titles
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n' \
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.4)
    return

# get classes
classes = img_datasets['test'].classes

# set images
n_images = 16

# plot incorrect
plot_predictions(incorrect_examples, classes, n_images)

# plot corect
plot_predictions(correct_examples, classes, n_images)
