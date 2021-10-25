# script for output of application
# connection between GUI and neural network, provides output of application
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

# code organization - import from other scripts
from appGUI import *
from appNetwork import *

