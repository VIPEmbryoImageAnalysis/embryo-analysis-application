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

import pytesseract
import glob

import threading

import matplotlib.pyplot as plt

# loading the model
model_config = {
    "input_height": 480,
    "input_width": 480,
    "n_classes": 3,
    "model_class": "unet"
    }

# initialize files list
files = []
#startingtime = []
percentthreshold = []
inputsize = []

# load latest weights
# hardcoded base path
base_path = os.path.abspath(".")
latest_weights = os.path.join(base_path, '1368weights.h5')
model = pretrained.model_from_checkpoint_path(model_config, latest_weights)

# Ask how much videos is being analyzed
numVideos = int(input('How many videos are you processing? '))

# Ask for file input, percent threshold, and initial size
# These files need to be inside the same folder
for x in range(1, numVideos+1):
    print('Video', x)
    n = input("What is the filepath? ")
    files += [n]
    z = input("What is the percent threshold? (only input number without %)")
    percentthreshold += [z]
    
    #cinitialSize = input("Do you want to input an initial size?")
    #if cinitialSize == "yes" or "Yes" or "YES" or "yES" or "yEs":
    y = input("What is the intial size? ") 
    inputsize += [y]

    

#define command: apply segmentation without overlay
def tempSeg(temp_path, temp_path2):
    model.predict_multiple(inp_dir=temp_path, out_dir=temp_path2, overlay_img=False)
            
#define command: apply segmentation with overlay
def visSeg(temp_path, temp_path3):
    model.predict_multiple(inp_dir=temp_path, out_dir=temp_path3, overlay_img=True)
   

# initialized more varaible and set inital values
org_folders = []
org2_folders = []
org3_folders = []
timeArray = []
numberoffiles = []
filelist = []
pixelArray = []
temppixelArray = []
pixeltoreal = 0.58 * 0.58
imsize = (480,480)
q = 1
tempvideocount = 0
initialsize =[]
finalsize =[]
finaltime = []
initialtime = []
growthsize = []
totaldata = []
totaldata1  = []
totaldata2 = []
df = []
df3 = []
df4 = []

# initialize working paths 
base_path1 = os.path.abspath(".")
homepath = os.getcwd()

#loop through each file in the file list
for j in files:
    timeArray = []
    temptimeArray = []
    temppixelArray = []
    pixelArray = []
    #determine the incremental value to update progress
    incremental = 60/len(files)
            
    #gets the filename
    filename = os.path.basename(j)
    filename = filename.replace('.avi','')

    #creates the images and segementation folders to store images
    temp_path = homepath + '/%s' % (filename) + 'Images/'
    os.mkdir(temp_path)
    temp_path2 = homepath + '/%s' % (filename) + 'Seg/'
    os.mkdir(temp_path2)
    temp_path3 = homepath + '/%s' % (filename) + 'Seg1/'
    os.mkdir(temp_path3)

    #store information about video into the list
    org_folders += [temp_path]
    org2_folders += [temp_path2]
    org3_folders += [temp_path3]
    filelist += [filename]
    #startingtimelogic = 0
            
    #print(org_folders)
    #print(org2_folders)

    #reads the video from the list
    video = cv2.VideoCapture(j)
    success,image = video.read()
    count = 1
    success = True
    while success:
        #changes the image to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #get the space between third and second digit
        digitSpace = image[475:489, 449:455]
                
        #convert CV2 image to black and white
        (thresh, blackAndWhiteImage) = cv2.threshold(digitSpace, 127, 255, cv2.THRESH_BINARY)

        #convert CV2 image to PIL image
        pdigitSpace = Image.fromarray(blackAndWhiteImage)
                
        #get colors present in image
        clrs = pdigitSpace.getcolors()
                
        #if there is only one color, there is only black
        if len(clrs) == 1:
                    
            #use two digit number w/ decimal
            currentTime = image[475:489, 456:483] 
                    
        else:
            #three digit number w/ decimal
            currentTime = image[475:489, 449:483]

        # Enchance image to get current time
        currentTime = Image.fromarray(currentTime)
        enhancer = ImageEnhance.Sharpness(currentTime)
        factor = 0.3
        currentTime = enhancer.enhance(factor)
        currentTime = np.asarray(currentTime)

        #debug statement
        #cv2.imshow("image", currentTime)
        #cv2.waitKey(0)
        #currentTime = cv2.resize(currentTime, (122,60))

        #path to tesseract.exe to get program to work
        #change to personal path
        #pytesseract_exe = os.path.join(base_path1, 'Tesseract-OCR/tesseract.exe')
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # use tesseracts' OCRfunction
        text = pytesseract.image_to_string(currentTime, lang='eng', config ='-c tessedit_char_whitelist=0123456789h')

        #debug statment
        #print(text)

        # convert the string recieved from OCR to int with numbers only
        M = int(''.join(filter(str.isdigit, text)))

        # convert to realtime displayed on time stamp and store it
        extractedTime = M / 10

        #debug statement
        #print(extractedTime)
        #print(startingtime[q-1])
    
        #if extractedTime == float(startingtime[q-1]):
        #    startingtimelogic = 1

        #image = cv2.resize(image, imsize)

        #if startingtimelogic == 1:

            #store extracted extracted time into an array
            #timeArray.append(extractedTime)

            #debug statement
            #print(timeArray)

            #resizes image, save new file, and reads in next video
            
            #cv2.imwrite(os.path.join(temp_path , "frame%d.png" % (count)), image)

            #increase count by 1
            #count += 1
            

            #debug statement
            #print(timeArray)

        timeArray.append(extractedTime)

        #resizes image, save new file, and reads in next video
        image = cv2.resize(image, imsize)
        cv2.imwrite(os.path.join(temp_path , "frame%d.png" % (count)), image)

        #increase count by 1
        count += 1

        success,image = video.read()
    #debug statement
    #print(count)
    #increases value of q by 1
    q += 1
                        
    #run images through the model
    tempSeg(temp_path, temp_path2)
    visSeg(temp_path, temp_path3)
            
            
    #debug statement
    #print(len(timeArray))

    #initialize i with value 1
    i = 1
    startingcountingsize = 0
    tempframenumber = []
    ok = 0
    #Loop through each video and count the amount of pixels from segmentation and store them
    while i < count:
        pixelcount = 0
        image = PIL.Image.open(os.path.join(temp_path2 , "frame%d.png" % (i)))
        for x in range(image.width):
            for y in range(image.width):
                if image.getpixel((x, y)) == (10, 34, 46):
                    pixelcount += 1
        if i == 1:
            #Get the scaling factor from the initial input size
            scalingfactor = float(inputsize[tempvideocount])/(pixelcount*pixeltoreal) 
        if i == 1:
            #debug statement
            #print(tempvideocount)
            #print(float(percentthreshold[tempvideocount]))

            #Gets the value for the threshold
            multiplicationthreshold = (float(percentthreshold[tempvideocount]) + 100)/100
            startingcountingsize = pixelcount * pixeltoreal * scalingfactor * multiplicationthreshold
        #Checks to see if the size passes the threshold
        if (pixelcount * pixeltoreal * scalingfactor) >= startingcountingsize:
            ok = 1
        #If the size passes the threshold
        if ok == 1:
            tempframenumber += [i]
            #pixelArray.append(pixelcount*pixeltoreal)
            temppixelArray.append(pixelcount*pixeltoreal*scalingfactor)
        
        #Store the pixel size to an array
        pixelArray.append(pixelcount*pixeltoreal*scalingfactor)
        i += 1
        #print(pixelcount*pixeltoreal)
        #print(scalingfactor)
        #print(pixelcount*pixeltoreal*scalingfactor)

    #Get the frames that only correspond to the values we want
    tempdeletenumber = tempframenumber[0] - 1
    temptimeArray = [None] * len(timeArray)
    print(temptimeArray)
    for i in range(0, len(timeArray)):    
        temptimeArray[i] = timeArray[i]
        
    print(temptimeArray)
    del temptimeArray[0:tempdeletenumber]
    print(temptimeArray)

    ok = 1
    i = 1
    #while i < count:
    #    if i == tempframenumber[0]:
    #        ok = 0
    #    if ok == 1:
    #        os.remove(os.path.join(temp_path , "frame%d.png" % (i)))
    #        os.remove(os.path.join(temp_path2 , "frame%d.png" % (i)))
    #        os.remove(os.path.join(temp_path3 , "frame%d.png" % (i)))
        #if i == tempframenumber[-1]:
        #    ok = 1
    #    i += 1

    #tempfilelist1 = os.listdir(temp_path)
    #tempfilelist2 = os.listdir(temp_path2)
    #tempfilelist3 = os.listdir(temp_path3)

    #filelist1 = natsorted(tempfilelist1, alg=ns.IGNORECASE)
    #filelist2 = natsorted(tempfilelist2, alg=ns.IGNORECASE)
    #filelist3 = natsorted(tempfilelist3, alg=ns.IGNORECASE)

    #i = 1
    #for file in filelist1:
    #    os.rename(os.path.join(temp_path, file), os.path.join(temp_path, "frame%d.png" % (i)))
    #    i += 1
    
    #i = 1
    #for file in filelist2:
    #    os.rename(os.path.join(temp_path2, file), os.path.join(temp_path2, "frame%d.png" % (i)))
    #    i += 1
    
    #i = 1
    #for file in filelist3:
    #    os.rename(os.path.join(temp_path3, file), os.path.join(temp_path3, "frame%d.png" % (i)))
    #    i += 1

    #saves number of files
    numberoffiles.append(len(timeArray))

    #debug statement
    #print(pixelArray)

    #initialize count back to 1
    count = 1
        
    #debug statement
    #print(numberoffiles)

    #initialize new values
    size = []
    time = []
    i = 0
    k = 0
    p = []

    #Run through each list to get the initial size, final size, and final time
    while k < numberoffiles[tempvideocount]:
        #If it is the first frame
        if k == 0:
            initialsize.append(pixelArray[i])
            initialtime.append(timeArray[i])
        #If it is the last frame
        if k == (numberoffiles[tempvideocount]-1):
            finalsize.append(pixelArray[i])
            finaltime.append(timeArray[i]-initialtime[tempvideocount])
        #Get all of the size and times into an array
        #size.append(pixelArray[i])
        #time.append(timeArray[i])
        #print(pixelArray[i])
        i += 1
        k += 1
        #print(i)
        #print(time)
        #print(size)
        #Debug Statement
        #print(time)
        #print(time[0])

    stoptime = 0

    #Check to see if time passes 10h after the threshold point
    timeholder = temptimeArray[0]
    for t in range(len(temptimeArray)):
        temptimeArray[t] = round(temptimeArray[t] - timeholder, 3)
        if temptimeArray[t] >= 10:
            if stoptime == 0:
                stoptime = t

    #delete the info not within the 10 h from the arrays
    if stoptime != 0:
        del temptimeArray[stoptime+1:-1]
        del temppixelArray[stoptime+1:-1]
        del temptimeArray[-1]
        del temppixelArray[-1]
    
    modtimeArray = timeArray
    #get modified time array for plot
    for t in range(len(modtimeArray)):
        modtimeArray[t] = round(modtimeArray[t] - timeholder, 3)


    #gets the average growth size
    z = np.polyfit(temptimeArray, temppixelArray, 1)
    p = np.poly1d(z)

    #debug statement
    #print(p)

    #add growth size to the list
    #print(p[1])
    growthsize.append(p[1])

    #debug statements
    #print(growthsize)
    #print(size)
    #print(time)
    #print(initialsize)
    #print(finalsize)

    #Get plotting info
    data2 = {'Time': modtimeArray, 'Embryo_Size': pixelArray}
    data1 = {'Time': temptimeArray, 'Embryo_Size': temppixelArray}
    data0 = {'Time': temptimeArray, 'Growth_Rate': p(temptimeArray)}

    totaldata.append(data1)
    totaldata1.append(data0)
    totaldata2.append(data2)

    #print(totaldata)
    #print(totaldata1)

    df1 = DataFrame(totaldata[tempvideocount],columns=['Time','Embryo_Size'])
    df2 = DataFrame(totaldata1[tempvideocount],columns=['Time','Growth_Rate'])
    df4 = DataFrame(totaldata2[tempvideocount],columns=['Time','Embryo_Size'])
    df.append(df4)
    df3.append(df2)
    

    #clears the size and time arrays
    #size.clear()
    #time.clear()
    k = 0

    tempvideocount += 1
    #sets i to 0
i = 0

# initialize base path 4
base_path4 = os.getcwd()
        
csvfile2 = os.path.join(base_path4, 'embryo_results.csv')

#print(filelist)
#print(growthsize)

# Generate empty CSV and stors info of the files in it
with open(csvfile2, 'w', newline='') as csvfile:
    fieldnames = ['Videos','Time (h)','Initial Size (um^2)','Final Size (um^2)',
                'Final Size Rank','Average Growth Rate (um^2/h)','Avg Growth Rate Rank']
    thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    thewriter.writeheader()

    #Writes the data from each list
    for j in filelist:
        #debug statements
        #print(j)
        #print(finaltime[i])
        #print(initialsize[i])
        #print(finalsize[i])
        #print(growthsize[i])

        thewriter.writerow({'Videos': j, 'Time (h)': finaltime[i], 'Initial Size (um^2)': initialsize[i], 
                            'Final Size (um^2)': finalsize[i], 'Final Size Rank': 1, 
                            'Average Growth Rate (um^2/h)': growthsize[i], 'Avg Growth Rate Rank': 1})
        i += 1

    #changes the rank so the are ranked by greatest growth rate and final size
if len(filelist) > 1:
    
        # Reopen the CSV file
    with open(csvfile2, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
     
        # Size Ranking
    tempSize = []
    
        #Gets size data
    for i in data[1:]:
        tempSize.append(float((i[3])))
    
    x = tempSize
    index = [0]*len(x)

        #reranks all of the videos
    for i in range(len(x)): 
        (r, s) = (1, 1) 
        for j in range(len(x)): 
            if j != i and x[j] > x[i]: 
                r += 1
            if j != i and x[j] == x[i]: 
                s += 1       
         
            # Use formula to obtain rank 
        index[i] = int(r + (s - 1) / 2)
    
        # Replace preset ranking with new ranking
    pos1 = 0
    for i in data[1:]:
        i[4] = str(index[pos1])
        pos1 = pos1 + 1
    
        # Growth Ranking
    tempGrowth = []
    for i in data[1:]:
        tempGrowth.append(int(float(i[5])))
    
    x = tempGrowth
    index = [0]*len(x)

        #reranks all of the videos
    for i in range(len(x)): 
        (r, s) = (1, 1) 
        for j in range(len(x)): 
            if j != i and x[j] > x[i]: 
                r += 1
            if j != i and x[j] == x[i]: 
                s += 1       
         
            # Use formula to obtain rank 
        index[i] = int(r + (s - 1) / 2)
        
        # Replace preset ranking with new ranking
    pos2 = 0
    for i in data[1:]:
            i[6] = str(index[pos2])
            pos2 = pos2 + 1
    
        # Rewrite list into CSV
    with open(csvfile2, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in data:
            wr.writerow(i)

    #Ranking Data for Size           
RankingDataSize = []            

    # Reopen the CSV file
with open(csvfile2, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
     
    #gets the data to display in the tables
for i in data[1:]:
    tempRankingData = []
    tempRank = float((i[4]))
    tempVideo = i[0]
    temp_FinalSize = i[3]
    tempRankingData.append(tempVideo)
    tempRankingData.append(tempRank)
    tempRankingData.append(temp_FinalSize)
    RankingDataSize.append(tempRankingData)

    #debug statement
    #print(RankingDataSize)

    #Ranking Data for Avg growth rate           
RankingDataGR = []            

    # Reopen the CSV file
with open(csvfile2, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
     
    #gets the data to display in the tables
for i in data[1:]:
    tempRankingData = []
    tempRank = float((i[6]))
    tempVideo = i[0]
    tempAverageGrowth = i[5]
    tempRankingData.append(tempVideo)
    tempRankingData.append(tempRank)
    tempRankingData.append(tempAverageGrowth)
    RankingDataGR.append(tempRankingData)

    #debug statement
    #print(RankingDataGR)

    #Current Video DAta           
CurrentVideoData = []            

    # Reopen the CSV file
with open(csvfile2, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
     
    #current video amount
CurrentTotalVideoAmount = 0

    #gets the data to display in the tables
for i in data[1:]:
    tempVideoData = []
    tempVideoName = i[0]
    tempFinalSize = float((i[3]))
    tempAverageGR = float((i[5]))
    tempVideoData.append(tempVideoName)
    tempVideoData.append(tempFinalSize)
    tempVideoData.append(tempAverageGR)
    CurrentVideoData.append(tempVideoData)
    CurrentTotalVideoAmount += 1

#Gets the plotting data for each video
base_path3 = os.getcwd()
k = 0
for j in filelist:
    dftemp = []
    test = []
    test2 = []
    csvfile3 = base_path3 + '/' + j + '_Data.csv'
    pd.concat([df[k], df3[k]], axis=1).to_csv(csvfile3)
    figure2 = plt.Figure(figsize=(5,4), dpi=100)
    ax2 = figure2.add_subplot(111)
    ax2.set_ylabel('Embryo Size (um^2)')
    minValuesObj = df[k].min()
    maxValuesObj = df[k].max()
    ax2.set_xlim([minValuesObj[0], maxValuesObj[0]])

    #Plot of Data
    dataLine = df[k]
    dataLine = dataLine[['Time','Embryo_Size']].groupby('Time').sum()
    dataLine.plot(kind='line', legend=True, ax=ax2, color='b',marker='o', fontsize=5)

    grLine = df3[k]
    grLine = grLine[['Time','Growth_Rate']].groupby('Time').sum()
    grLine.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=5)
    
    ax2.set_xlabel('Time (h)')
    ax2.set_title('Embryo Growth Over Time')
    
    ax2.axvline(0, color='k', linestyle='--')
    ax2.axhline(temppixelArray[0], color='k', linestyle='--')

    k += 1
    
ax2.figure

# plt.plot(timeArray,pixelArray)
# plt.xlabel("Time(h)")
# plt.ylabel("Embryo Size (um^2)")
# plt.title("Embryo Growth Over Time")
# plt.show()