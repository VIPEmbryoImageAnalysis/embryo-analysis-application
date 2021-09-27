import os
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file
from shutil import rmtree
import cv2

import pandas as pd

from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import PIL
from PIL import Image
import numpy as np
import csv
from keras_segmentation import pretrained
from array import *

import tensorflow
import pytesseract
import glob

#C:\Users\corpu\Desktop\MicroVIP Embryo\image-segmentation-keras-master\D-ArakakiNwells_1.10_video.avi
# loading the model
model_config = {
    "input_height": 480,
    "input_width": 480,
    "n_classes": 3,
    "model_class": "unet"
    }

# initialize files list
files = []
inputsize = [] #

# load latest weights
# hardcoded base path
base_path = os.path.abspath(".")
latest_weights = os.path.join(base_path, '1368weights.h5')
model = pretrained.model_from_checkpoint_path(model_config, latest_weights)

# Ask how much videos is being analyzed
numVideos = int(input('How many videos are you processing? '))

# Ask for file input
# These files need to be inside the same folder
for x in range(1, numVideos+1):
    print('Video', x)
    n = input("What is the filepath? ")
    files += [n]
    z = input("What is the intial size? ") 
    inputsize += [z]
    
    
# Ask for threshold percent
percentThreshold = int(input('Please specify the percent threshold to be used for average growth rate calculation. '))

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
pixeltoreal = 0.58 * 0.58
imsize = (480,480)
q = 1
tempvideocount = 0

# initialize working paths 
base_path1 = os.path.abspath(".")
homepath = os.getcwd()



#loop through each file in the file list
for j in files:            
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

        #debug statement
        #cv2.imshow("image", currentTime)
        #cv2.waitKey(0)
        #currentTime = cv2.resize(currentTime, (122,60))

        #path to tesseract.exe to get program to work
        #change to personal path
        #pytesseract_exe = os.path.join(base_path1, 'Tesseract-OCR/tesseract.exe')
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        #pytesseract.pytesseract.tesseract_cmd = pytesseract_exe

        # use tesseracts' OCRfunction
        text = pytesseract.image_to_string(currentTime, lang='eng', config ='-c tessedit_char_whitelist=0123456789h')

        #debug statment
        #print(text)

        # convert the string recieved from OCR to int with numbers only
        M = int(''.join(filter(str.isdigit, text)))

        # convert to realtime displayed on time stamp and store it
        extractedTime = M / 10

        #debug statement
        #print(M)

        #store extracted extracted time into an array
        timeArray.append(extractedTime)

        #resizes image, save new file, and reads in next video
        image = cv2.resize(image, imsize)
        cv2.imwrite(os.path.join(temp_path , "frame%d.png" % (count)), image)
        success,image = video.read()

        #increase count by 1
        count += 1
            

        #debug statement
        #print(timeArray)

        #increases value of q by 1
        q += 1

        #saves number of files
        numberoffiles.append(count - 1)
                        
        #run images through the model
        tempSeg(temp_path, temp_path2)
        visSeg(temp_path, temp_path3)
            
            
        #debug statement
        #print(len(timeArray))

        #initialize i with value 1
        i = 1

    #Loop through each video and count the amount of pixels from segmentation and store them
    while i < count:
        pixelcount = 0
        image = PIL.Image.open(os.path.join(temp_path2 , "frame%d.png" % (i)))
        for x in range(image.width):
            for y in range(image.width):
                if image.getpixel((x, y)) == (10, 34, 46):
                    pixelcount += 1
      #Checking for first frame of the video, finding scaling factor between pixel counted and the input size used
        if i == 1:
            scalingfactor = float(inputsize[tempvideocount])/(pixelcount*pixeltoreal) 
            
            
        #stores total size from multiplication into an array   
        pixelArray.append(pixelcount*pixeltoreal*scalingfactor)
        i += 1

    #debug statement
    #print(pixelArray)

    #initialize count back to 1
    count = 1
        
    #debug statement
    #print(numberoffiles)

    #initialize new values
    size = []
    time = []
    totaldata = []
    totaldata1  = []
    df = []
    df3 = []
    i = 0
    k = 0
    initialsize =[]
    finalsize =[]
    finaltime = []
    initialtime = []
    p = []
    growthsize =[]
        
        
        
    # Replace with Time Zero Average Growth Rate
    #Run through each list to get the initial size, final size, and final time
    
    while k <= (len(numberoffiles) - 1):
        if k == 0:
            initialsize.append(pixelArray[i])
            initialtime.append(timeArray[i])
                    
        if k == (len(numberoffiles) - 1):
            finalsize.append(pixelArray[i])
            finaltime.append(timeArray[i])
                    
        #Get all of the size and times into an array
        size.append(pixelArray[i])
        time.append(timeArray[i])
        i = i + 1
        k = k + 1
                
    timeholder = time[0]
    for t in range(len(time)):
        time[t] = round(time[t] - timeholder, 3)
        
    print(size)
    print(time)
            
            
    # index variable s
    s = 0
            
    #size = [23000,23400,25000,27000,28700,30000,31000,32000,33000,33500,36000]
            
    #percentThreshold = 10
            
    while (s < len(size)) and (percentThreshold > ((size[s] - size[0])/abs(size[0]) * 100)): 
        s = s + 1


    # special time zero arrays
    timezeroTimes = []
    timezeroSizes = []
    while s < len(size):
        timezeroTimes.append(time[s])
        timezeroSizes.append(size[s])
        s = s + 1
        
    print(timezeroTimes)
    print(timezeroSizes)
            
    #gets the average growth size
    z = np.polyfit(timezeroTimes, timezeroSizes, 1)
    p = np.poly1d(z)

    #debug statement
    #print(p)

    #add growth size to the list
    growthsize.append(p[1])

    #debug statements
    #print(growthsize)
    #print(size)
    #print(time)
    #print(initialsize)
    #print(finalsize)

    #Get plotting info
    #data1 = {'Time': time, 'Embryo_Size': size}
    #data0 = {'Time': time, 'Growth_Rate': p(time)}
            
    #totaldata.append(data1)
    #totaldata1.append(data0)
    #df1 = DataFrame(totaldata[x],columns=['Time','Embryo_Size'])
    #df2 = DataFrame(totaldata1[x],columns=['Time','Growth_Rate'])
    #df.append(df1)
    #df3.append(df2)

    #clears the size and time arrays
    size.clear()
    time.clear()
    k = 0
    #sets i to 0
    i = 0

    # initialize base path 4
    base_path4 = os.getcwd()
        
    csvfile2 = os.path.join(base_path4, 'embryo_results.csv')
        
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
                            'Final Size (um^2)': finalsize[i], 'Final Size Rank': 1, 'Average Growth Rate (um^2/h)': growthsize[i], 'Avg Growth Rate Rank': 1})
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
    tempvideocount += 1
