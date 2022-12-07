import copy
import sys
from os import path
import os
import re
import pandas as pd
import numpy as np
from numpy import save
import cv2
import time
from tqdm import tqdm
import tqdm.notebook as tq
import random
import glob
import math
import os, time, gc, random, warnings, joblib
import warnings
import itertools
import argparse
from PIL import Image
#import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from  builtins import any as b_any
from collections import OrderedDict
from prettytable import PrettyTable
from tabulate import tabulate

import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda import amp
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import ConcatDataset

import timm
import torchvision
from torchvision import datasets, models, transforms

import fastai
from fastai import * 
from fastai.vision import *

from sklearn.model_selection import train_test_split
from collections import defaultdict

from imgaug import augmenters as iaa

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import precision_score,f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pretrainedmodels
from efficientnet_pytorch import EfficientNet
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
