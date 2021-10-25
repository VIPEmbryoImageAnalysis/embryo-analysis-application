# script for neural network related functions of the application
# import on app-GUI.py for functionality

import os
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file
from shutil import rmtree
import cv2
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
import pandas as pd
from tkinter import filedialog, Tk, HORIZONTAL
from tkinter.ttk import Combobox, Progressbar
from tkinter import messagebox
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PIL
from PIL import Image, ImageEnhance
import numpy as np
import csv
import tensorflow
from tensorflow import keras
from keras_segmentation import pretrained
from array import *
from ttkbootstrap import Style

import pytesseract
import glob

import threading
import mttkinter

# loading the model
model_config = {
    "input_height": 480,
    "input_width": 480,
    "n_classes": 3,
    "model_class": "unet"
    }

# load latest weights
#change to path so not dependent on computer folder
try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = sys._MEIPASS
except Exception:
    base_path = os.path.abspath(".")
latest_weights = os.path.join(base_path, '1368weights.h5')
model = pretrained.model_from_checkpoint_path(model_config, latest_weights)
