# LIVE VERSION OF APPLICATION
# DO NOT TEST USING THIS SCRIPT

# DEPLOYED DATE: 11/2/2021

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
import ttkbootstrap
# import styling for application theme
from ttkbootstrap import Style
import mttkinter

# declare tkinter theme
style = Style(theme='litera')
#style.configure('TLabel', font=('Helvetica', 12))

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

#application window
#root = tk.Tk()
root=style.master
root.title("Embryo Analysis")
icon = os.path.join(base_path, 'embryoIcon.ico')
root.iconbitmap(icon)
root.geometry("900x600")

#initialize variables to keep track of video name list and values for Initial Conditions
videoTitleList=[]
timeZeroList=[]
embryoSizeList=[]

# define event: open file explorer, select video, display selected video in table
def uploadVideos():
    filepath = tk.filedialog.askopenfilenames(filetypes=[("AVI", ".avi"),
        ("MOV", ".mov"), ("MP4", ".mp4"), ("FLV", ".flv")])
    i=0
    while i<len(filepath):
        displayFiles1.insert('', 'end', text= filepath[i], values=(filepath[i]))
        i += 1

    #gets the name of the video and appends it to a list, also append 0's to
    #timeZeroList and embryoSizeList as default values for initialConditions function
    videoPaths = (list(filepath))
    for i in videoPaths:
        name = i.split("/")
        vidName = name[len(name)-1]
        videoTitleList.append(vidName)
        timeZeroList.append("6")
        embryoSizeList.append("0")

#define event: removes all of the videos in the file list 
def RemoveAllVideos():
    for child in displayFiles1.get_children():
        displayFiles1.delete(child)
    #clears lists used in initialConditions
    videoTitleList.clear()
    embryoSizeList.clear()
    timeZeroList.clear()

# define event: remove selected file
def removeFile(): 
    try:
        #gets filepath for selected video and splices it into a list by '/'
        item=displayFiles1.item(displayFiles1.focus())
        videoPath=item['text'].split('/')

        selected_item1 = displayFiles1.selection()[0] # get selected item
        displayFiles1.delete(selected_item1)

        #get video name at the end of the list and removes it from videoTitleList
        videoPathName=videoPath[len(videoPath)-1]
        vidIndex=videoTitleList.index(videoPathName)
        videoTitleList.remove(videoPathName)
        #deletes the initialConditions value for the selected video at its respective index
        del embryoSizeList[vidIndex]
        del timeZeroList[vidIndex]


    except IndexError:
        messagebox.showerror("Error", "No selected file") # error message

# define event: remove the directories created for images and segmentations    
def removeTemporaryDirs(org_folders, org2_folders, org3_folders, csvfile2):
    for i in org_folders:
        rmtree(i)
    
    for i in org2_folders:
        rmtree(i)
    
    for i in org3_folders:
        rmtree(i)

    os.remove(csvfile2)

#define event: allows user to set initial conditions (Time Zero and Initial Embryo Size)
def initialConditions():

    #check to see if videos are uploaded
    if len(videoTitleList)!=0:
        newWindow = tk.Toplevel(root)
        newWindow.title("Edit Initial Conditions")
        newWindow.geometry("500x200")

        #configures window and shows video titles in window with scrollbar
        scrollbar = Scrollbar(newWindow)
        videoListbox = Listbox(newWindow,yscrollcommand=scrollbar.set)
        videoListbox.pack(side=LEFT,fill=BOTH)
        videoListbox.config(width=35)
        scrollbar.pack(side=LEFT,fill=Y)
        scrollbar.config(command=videoListbox.yview)

        backButton = Button(newWindow, text='Done', font = ("Helvetica", 11),command=newWindow.destroy)
        backButton.place(relx=0.87, rely=0.8)
        backButton['bg'] = '#3FA8B5'
        
        selectedLabel = Label(newWindow,text='')
        selectedLabel.place(relx=.55)
        for i in videoTitleList:
            videoListbox.insert(END,i)

    #When user clicks on a video and clicks "Select", it will bring up a window that allows input for initial conditions
    def select():
        selectedVid = videoListbox.get(ANCHOR)
        if selectedVid:
            vidIndex = videoTitleList.index(selectedVid)
            selectedLabel.config(text=selectedVid)
            popupWindow = tk.Toplevel(root)
            popupWindow.title("Configure Data")
            popupWindow.geometry("450x200")

            #After user enters initial condtions and clicks "Done" this function will check and append data to embryoSizeList and timeZeroList at its
            #respective index relative to the video title
            def getInputValues():
                def check_float(potential_float):
                    try:
                        float(potential_float)
                        return True
                    except ValueError:
                        return False
                        
                if check_float(timeZeroInput.get()):
                    timeZeroList[vidIndex]=timeZeroInput.get()
   
                if check_float(embryoSizeInput.get()):
                    embryoSizeList[vidIndex]=embryoSizeInput.get()

            #display Initial Embryo Size label and entry box
            initialEmbryoSizeText = ttk.Label(popupWindow, text = "Initial Embryo Size (um^2):", font = ("Helvetica", 11))
            initialEmbryoSizeText.place(relx=0.1, rely=0.2)

            embryoSizeInput = tk.Entry(popupWindow, width=15)
            embryoSizeInput.place(relx=0.5, rely=0.2)

            #display Time Zero Label and entry box
            timeZeroText = ttk.Label(popupWindow, text = "Percent Threshold (%):", font = ("Helvetica", 11))
            timeZeroText.place(relx=0.1, rely=0.4)

            timeZeroInput = tk.Entry(popupWindow, width=15)
            timeZeroInput.place(relx=0.5, rely=0.4)

            doneButton = tk.Button(popupWindow,text="Done", font = ("Helvetica", 11), width=8, command=lambda:[getInputValues(),popupWindow.destroy()])
            doneButton.place(relx=.75,rely=.7)
            doneButton['bg'] = '#3FA8B5'

    try:
        selectButton = Button(newWindow, text = 'Select', font = ("Helvetica", 11), command=select)
        selectButton.place(relx=0.75, rely=0.8)
        selectButton['bg'] = '#3FA8B5'

    except Exception:
        None


#define event: Results window that contains all of the results
def openNewWindow():
    try:
     
        #define event: Gets the current video number used for diaplying the current graph and data
        def GetCurrentNumber():
            
            #Current Video Data           
            CurrentVideoNumber = []            

            try:
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path2 = sys._MEIPASS
            except Exception:
                base_path2 = os.path.abspath(".")

            csvfile1 = os.path.join(base_path2, 'embryo_results.csv')

            # Reopen the CSV file
            with open(csvfile1, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
            
            #get all of the data needed from the csv file (filenames)
            for i in data[1:]:
                tempVideoData = []
                tempVideoName = i[0]
                tempVideoData.append(tempVideoName)
                CurrentVideoNumber.append(tempVideoData)
            
            #debug statement
            #print(CurrentVideoNumber)
            
            #gets the current video information of the video showing
            child = tree3.get_children()
            videoname = tree3.item(child)["values"][0]

            #Gets the number that corresponds to the current video showed
            for l in range(len(CurrentVideoNumber)):
                #debug statement
                #print(l)
                #print(CurrentVideoNumber[l])

                #sets variable videonumber1 to the video number value
                if CurrentVideoNumber[l][0] == videoname:
                    videonumber1 = l
            
            #debug statement
            #print(videonumber1)

            #returns the video number
            return videonumber1

        #define event: displays the next video data on the screen when called
        def NextVideo(CurrentTotalVideoAmount, CurrentVideoData, df, df3, newWindow, frame, tempPixelArray2):
            #Calls function to get current videos number
            CurrentVideoNumber = GetCurrentNumber()

            #Next video is one above the current number but if video is last in list set number to the first video number
            NextVideoNumber = CurrentVideoNumber + 1
            if NextVideoNumber == CurrentTotalVideoAmount:
                NextVideoNumber = 0
            
            #Get data for the next video
            CurrentVideoName = CurrentVideoData[NextVideoNumber][0]
            CurrentFinalSize = CurrentVideoData[NextVideoNumber][1]
            CurrentGR = CurrentVideoData[NextVideoNumber][2]

            #remove the data in the current video table
            for child in tree3.get_children():
                tree3.delete(child)

            #inserts new image into the current video table
            tree3.insert("", "end", text="0", values=(CurrentVideoName,CurrentFinalSize,CurrentGR))

            #removes current graph
            frame.destroy()

            #creates new graph
            frame = tk.Frame(newWindow)
            figure2 = plt.Figure(figsize=(5,4), dpi=100)
            ax2 = figure2.add_subplot(111)
            ax2.set_ylabel('Embryo Size (um^2)')
            minValuesObj = df[NextVideoNumber].min()
            maxValuesObj = df[NextVideoNumber].max()
            ax2.set_xlim([minValuesObj[0], maxValuesObj[0]])

            #Plot of Data
            dataLine = df[NextVideoNumber]
            dataLine = dataLine[['Time','Embryo_Size']].groupby('Time').sum()
            dataLine.plot(kind='line', legend=True, ax=ax2, color='b',marker='o', fontsize=5)

            grLine = df3[NextVideoNumber]
            grLine = grLine[['Time','Growth_Rate']].groupby('Time').sum()
            grLine.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=5)
            
            ax2.set_xlabel('Time (h)')
            ax2.set_title('Embryo Growth Over Time')
            
            ax2.axvline(0, color='k', linestyle='--')
            ax2.axhline(tempPixelArray2[NextVideoNumber], color='k', linestyle='--')

            line2 = FigureCanvasTkAgg(figure2, newWindow)
            line2.get_tk_widget().place(relx = 0.03, rely = 0.05, relwidth=0.57, relheight=0.6)

        #define event:  displays the previous video data on the screen when called
        def BackVideo(CurrentTotalVideoAmount, CurrentVideoData, df, df3, newWindow, frame, tempPixelArray2):
            #Calls function to get current videos number
            CurrentVideoNumber = GetCurrentNumber()

            #If the video is the first in the list next video is the last video
            LastVideoNumber = CurrentVideoNumber - 1
            if LastVideoNumber == - 1:
                LastVideoNumber = CurrentTotalVideoAmount - 1

            #Get data for the last video
            CurrentVideoName = CurrentVideoData[LastVideoNumber][0]
            CurrentFinalSize = CurrentVideoData[LastVideoNumber][1]
            CurrentGR = CurrentVideoData[LastVideoNumber][2]

            #remove the data in the current video table
            for child in tree3.get_children():
                tree3.delete(child)

            #inserts new image into the current video table
            tree3.insert("", "end", text="0", values=(CurrentVideoName,CurrentFinalSize,CurrentGR))

            #removes current graph
            frame.destroy()

            #creates new graph
            frame = tk.Frame(newWindow)
            figure2 = plt.Figure(figsize=(5,4), dpi=100)
            ax2 = figure2.add_subplot(111)
            ax2.set_ylabel('Embryo Size (um^2)')
            minValuesObj = df[LastVideoNumber].min()
            maxValuesObj = df[LastVideoNumber].max()
            ax2.set_xlim([minValuesObj[0], maxValuesObj[0]])

            #Plot of Data
            dataLine = df[LastVideoNumber]
            dataLine = dataLine[['Time','Embryo_Size']].groupby('Time').sum()
            dataLine.plot(kind='line', legend=True, ax=ax2, color='b',marker='o', fontsize=5)

            grLine = df3[LastVideoNumber]
            grLine = grLine[['Time','Growth_Rate']].groupby('Time').sum()
            grLine.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=5)
            
            ax2.set_xlabel('Time (h)')
            ax2.set_title('Embryo Growth Over Time')
            
            ax2.axvline(0, color='k', linestyle='--')
            ax2.axhline(tempPixelArray2[LastVideoNumber], color='k', linestyle='--')

            line2 = FigureCanvasTkAgg(figure2, newWindow)
            line2.get_tk_widget().place(relx = 0.03, rely = 0.05, relwidth=0.57, relheight=0.6)
        
        #define event: ask user where to save CSV
        def saveResults(df, df3, filelist, tempPixelArray2):
            dest = filedialog.askdirectory()
            #print(dest)
            try:
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path3 = sys._MEIPASS
            except Exception:
                base_path3 = os.getcwd()
            path = base_path3 + '/embryo_results.csv'
            copy_file(path, dest)
            k = 0
            for j in filelist:
                csvfile3 = dest + '/' + j + '_Data.csv'
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
                ax2.axhline(tempPixelArray2[k], color='k', linestyle='--')
                figname = dest + '/' + j + '_Plot.png'
                figure2.savefig(figname)
                k += 1

        #define event: ask user for folder to save images in
        def savePics(org_folders, org3_folders, filelist):
            dest = filedialog.askdirectory()
            flag = 0
            flag2 = 0
            
            #copy from source image folders to specified folder
            for Path in org_folders:
                copy_tree(Path, (dest + '/%s' % (filelist[flag]) + '_Images/'))
                flag += 1
            
            #copy from source segmentation folders to specified folder
            for Path2 in org3_folders:
                copy_tree(Path2, (dest + '/%s' % (filelist[flag2]) + '_Segmentations/'))
                flag2 += 1
        
        #define command: apply segmentation without overlay
        def tempSeg(temp_path, temp_path2):
            model.predict_multiple(inp_dir=temp_path, out_dir=temp_path2, overlay_img=False)
            
        #define command: apply segmentation with overlay
        def visSeg(temp_path, temp_path3):
            model.predict_multiple(inp_dir=temp_path, out_dir=temp_path3, overlay_img=True)

        #creates the loading bar window
        loadingWindow = tk.Toplevel(root)
        loadingWindow.title("Please Wait For Results")
        loadingWindow.geometry("300x100")
        #style='primary.Striped.Horizontal.TProgressbar'
        progress = Progressbar(loadingWindow, orient = HORIZONTAL, length=100, mode = 'determinate')
        progress.pack(pady = 10)
        old = 20

        #creates the analyzed window
        newWindow = tk.Toplevel(root)
        newWindow.title("Analyze Results")
        newWindow.geometry("1480x750")
        

        #initialized files list
        files = []

        #gets all of the files in the list
        for child in displayFiles1.get_children():
            path = displayFiles1.item(child)["text"]
            files.append(path)
        
        #get number of videos
        numVideos = len(files)

        #initialized more varaible and set inital values
        org_folders = []
        org2_folders = []
        org3_folders = []
        numberoffiles = []
        filelist = []
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
        tempPixelArray2 = []
        #store information into threshhold and input size arrays
        percentthreshold = timeZeroList
        inputsize = embryoSizeList

        
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS 
            base_path1 = sys._MEIPASS
        except Exception:
            base_path1 = os.path.abspath(".")

        #get current homepath
        #homepath = os.getcwd()

        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            homepath = sys._MEIPASS
        except Exception:
            homepath = os.getcwd()

        #debug statement
        #print(files)
        
        #update the progress bar
        progress['value'] = old
        loadingWindow.update()

        #loop through each file in the file list
        for j in files:
            timeArray = []
            pixelArray = []
            temptimeArray = []
            temppixelArray = []
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
                    currentTime = image[475:489, 456:481] 
                    
                else:
                    #three digit number w/ decimal
                    currentTime = image[475:489, 449:481]


                #sharpen image if 0.3 doesn't work try change factor to 0.4
                currentTime = Image.fromarray(currentTime)
                enhancer = ImageEnhance.Sharpness(currentTime)
                factor = 0.3047
                currentTime = enhancer.enhance(factor)
                currentTime = np.asarray(currentTime)

                #debug statement
                #cv2.imshow("image", currentTime)
                #cv2.waitKey(0)
                #currentTime = cv2.resize(currentTime, (122,60))

                #path to tesseract.exe to get program to work
                #change to personal path
                pytesseract_exe = os.path.join(base_path1, 'Tesseract-OCR/tesseract.exe')
                pytesseract.pytesseract.tesseract_cmd = pytesseract_exe

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
                

                #increase count by 1
                count += 1

                success,image = video.read()
            

            #debug statement
            #print(timeArray)

            #increases value of q by 1
            q += 1

            #saves number of files
            numberoffiles.append(count - 1)
                        
            #run images through the model
            # apply segmentation with multithreading
            # replace inp_dir with folder directory
            # replace out_dir with new folder directory
            thread1 = threading.Thread(target=tempSeg, args=(temp_path, temp_path2))
            thread2 = threading.Thread(target=visSeg, args=(temp_path, temp_path3))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            
            #update current progress after each segmentation
            progress['value'] = incremental + old
            loadingWindow.update()
            old = incremental + old
            
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
                    if float(inputsize[tempvideocount]) == 0:
                        scalingfactor = 1
                    else:
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

            #debug statement
            #print(pixelArray)
            tempdeletenumber = tempframenumber[0] - 1
            temptimeArray = [None] * len(timeArray)
            #print(temptimeArray)
            for i in range(0, len(timeArray)):    
                temptimeArray[i] = timeArray[i]
                
            #print(temptimeArray)
            del temptimeArray[0:tempdeletenumber]
            #print(temptimeArray)

            ok = 1
            i = 1
            #initialize count back to 1
            count = 1
            size = []
            time = []
            i = 0
            k = 0
            p = []
            #debug statement
            #print(numberoffiles)

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
                if stoptime != (len(timeArray)-1):
                    del temptimeArray[-1]
                    del temppixelArray[-1]
            
            modtimeArray = timeArray
            #get modified time array for plot
            for t in range(len(modtimeArray)):
                modtimeArray[t] = round(modtimeArray[t] - timeholder, 3)


            #gets the average growth size
            z = np.polyfit(temptimeArray, temppixelArray, 1)
            p = np.poly1d(z)
            tempPixelArray2.append(temppixelArray[0])

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

        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path4 = sys._MEIPASS
        except Exception:
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

        #debug statement
        #print(CurrentVideoData)

        #Sorting Ranking Data
        RankingDataSize.sort(key=lambda e: e[1])
        RankingDataGR.sort(key=lambda e: e[1])
        
        #last update and noti sound when complete
        progress['value'] = 100
        loadingWindow.update()
        loadingWindow.bell()
        loadingWindow.after(500, loadingWindow.destroy())

        #Plot of Data
        frame = tk.Frame(newWindow)
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111)
        ax2.set_ylabel('Embryo Size (um^2)')
        line2 = FigureCanvasTkAgg(figure2, newWindow)
        line2.get_tk_widget().place(relx = 0.03, rely = 0.05, relwidth=0.57, relheight=0.6)
        minValuesObj = df[0].min()
        maxValuesObj = df[0].max()
        ax2.set_xlim([minValuesObj[0], maxValuesObj[0]])

        #Plot of Data
        dataLine = df[0]
        dataLine = dataLine[['Time','Embryo_Size']].groupby('Time').sum()
        dataLine.plot(kind='line', legend=True, ax=ax2, color='b',marker='o', fontsize=5)

        grLine = df3[0]
        grLine = grLine[['Time','Growth_Rate']].groupby('Time').sum()
        grLine.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=5)
        
        ax2.set_xlabel('Time (h)')
        ax2.set_title('Embryo Growth Over Time')
        
        ax2.axvline(0, color='k', linestyle='--')
        ax2.axhline(tempPixelArray2[0], color='k', linestyle='--')

        #Tables for Ranking Data
        frame2 = tk.Frame(newWindow)
        frame2.place(relx = 0.625, rely = 0.05, relwidth=0.325, relheight=0.3)
        label = tk.Label(frame2, text="Size Ranking")
        label.pack(expand=YES)
        #style='secondary.Treeview'
        tree1 = ttk.Treeview(frame2, style='info.Treeview', columns=("Video", "Rank", "Final Size (um^2)"))
        tree1.pack(expand=YES, fill=BOTH)
        tree1["columns"] = ("1", "2", "3")
        tree1['show'] = 'headings'
        tree1.column("1", minwidth=0, width=130)
        tree1.column("2", minwidth=0, width=100)
        tree1.column("3", minwidth=0, width=180)
        tree1.heading("1", text="Video")
        tree1.heading("2", text="Rank")
        tree1.heading("3", text="Final Size (um^2)")
        number = 0
        for rank in RankingDataSize:
            SizeRankingVideo = rank[0]
            SizeRankingRank = int(rank[1])
            SizeRankingSize = rank[2]
            #print(SizeRankingVideo)
            #print(SizeRankingRank)
            tree1.insert("", "end", text=number, values=(SizeRankingVideo,SizeRankingRank, SizeRankingSize))
            number += 1

        #Tables for Ranking Data
        frame3 = tk.Frame(newWindow)
        frame3.place(relx = 0.625, rely = 0.40, relwidth=0.325, relheight=0.3)
        label = tk.Label(frame3, text="Average Growth Rate Ranking")
        label.pack(expand=YES)
        #style='secondary.Treeview'
        tree2 = ttk.Treeview(frame3, style='info.Treeview', columns=("Video", "Rank", "Average Growth Rate (um^2/h)"))
        tree2.pack(expand=YES, fill=BOTH)
        tree2["columns"] = ("1", "2", "3")
        tree2['show'] = 'headings'
        tree2.column("1", minwidth=0, width=130)
        tree2.column("2", minwidth=0, width=100)
        tree2.column("3", minwidth=0, width=180)
        tree2.heading("1", text="Video")
        tree2.heading("2", text="Rank")
        tree2.heading("3", text="Average Growth Rate (um^2/h)")
        number = 0
        for rank in RankingDataGR:
            GRRankingVideo = rank[0]
            GRRankingRank = int(rank[1])
            GRRankingRate = rank[2]
            #print(GRRankingVideo)
            #print(GRRankingRank)
            tree2.insert("", "end", text=number, values=(GRRankingVideo,GRRankingRank, GRRankingRate))
            number += 1

        #Table for current data and graph for current data
        frame4 = tk.Frame(newWindow)
        frame4.place(relx = 0.15, rely = 0.75, relwidth=0.375, relheight=0.118)
        label = tk.Label(frame4, text="Current Video")
        label.pack(expand=YES)
        #style='secondary.Treeview'
        tree3 = ttk.Treeview(frame4, style='info.Treeview', columns=("Video", "Final Size (um^2)", "Average Growth Rate (um^2/h)"))
        tree3.pack(expand=YES, fill=BOTH)
        tree3["columns"] = ("1", "2", "3")
        tree3['show'] = 'headings'
        tree3.column("1", minwidth=0, width=130)
        tree3.column("2", minwidth=0, width=140)
        tree3.column("3", minwidth=0, width=180)
        tree3.heading("1", text="Video")
        tree3.heading("2", text="Final Size (um^2)")
        tree3.heading("3", text="Average Growth Rate (um^2/h)")
        CurrentVideoName = CurrentVideoData[0][0]
        CurrentFinalSize = CurrentVideoData[0][1]
        CurrentGR = CurrentVideoData[0][2]
        #print(CurrentVideoName)
        #print(CurrentFinalSize)
        #print(CurrentGR)
        tree3.insert("", "end", text="0", values=(CurrentVideoName,CurrentFinalSize,CurrentGR))

        #Button to go to the last value
        #style='Outline.TButton'
        LeftButton = ttk.Button(newWindow, text = '<', style='secondary.TButton', command=lambda:BackVideo(CurrentTotalVideoAmount, CurrentVideoData, df, df3, newWindow, frame, tempPixelArray2))
        LeftButton.place(relx=0.13, rely=0.783, relwidth=0.02, relheight=0.085)

        #Button to go to the next value
        #style='Outline.TButton'
        RightButton = ttk.Button(newWindow, text = '>', style='secondary.TButton', command=lambda:NextVideo(CurrentTotalVideoAmount, CurrentVideoData, df, df3, newWindow, frame, tempPixelArray2))
        RightButton.place(relx=0.525, rely=0.783, relwidth=0.02, relheight=0.085)

        #creates new window to hold buttons
        frame5 = tk.Frame(newWindow)
        frame5.place(relx = 0.65, rely = 0.75, relwidth=0.3, relheight=0.2)

        #creates buttons for next steps
        #style='Outline.TButton'
        yesRemoveButton1 = ttk.Button(frame5, text = 'Save Images and Labels', style='info.TButton', command = lambda: savePics(org_folders, org3_folders, filelist))
        yesRemoveButton1.place(relx=0.075, rely=0.1, relwidth=0.40, relheight=0.30)
        #style='Outline.TButton'
        yesRemoveButton2 = ttk.Button(frame5, text = 'Save Results', style='info.TButton', command = lambda: saveResults(df, df3, filelist, tempPixelArray2))
        yesRemoveButton2.place(relx=0.525, rely=0.1, relwidth=0.40, relheight=0.30)
        #style='Outline.TButton'
        yesRemoveButton3 = ttk.Button(frame5, text = 'Load new videos', style='info.TButton', command = lambda: [newWindow.destroy(), RemoveAllVideos(), removeTemporaryDirs(org_folders, org2_folders, org3_folders, csvfile2)])
        yesRemoveButton3.place(relx=0.075, rely=0.6, relwidth=0.40, relheight=0.30)
        #style='Outline.TButton'
        yesRemoveButton4 = ttk.Button(frame5, text = 'Close App', style='info.TButton', command = lambda: [newWindow.destroy(), root.destroy(), removeTemporaryDirs(org_folders, org2_folders, org3_folders, csvfile2)])
        yesRemoveButton4.place(relx=0.525, rely=0.6, relwidth=0.40, relheight=0.30)

        def on_closing():
            removeTemporaryDirs(org_folders, org2_folders, org3_folders, csvfile2)
            RemoveAllVideos()
            newWindow.destroy()
        
        newWindow.protocol("WM_DELETE_WINDOW", on_closing)
        
    except IndexError:
        messagebox.showerror("Error", "No uploaded files")  # error message

#create initialConditions Button
initialConditionsButton = ttk.Button(root, text = 'Edit Initial Conditions', style='info.Outline.TButton', command=initialConditions)
initialConditionsButton.place(relx=.635,rely=.2)

# button: upload videos
uploadButton = ttk.Button(root, text = 'Upload Video', style='info.TButton', command=uploadVideos)
uploadButton.place(relx=0.26, rely=0.2, anchor='n')

# display selected files  words
filesText = ttk.Label(root, text = "Uploaded files:", font = ("Helvetica", 11))  # label for selected files
filesText.place(relx = 0.07, rely = 0.3)

# display file path
frame1 = tk.Frame(root)
frame1.place(relx = 0.2, rely = 0.3, relwidth=0.6, relheight=0.4)
displayFiles1 = ttk.Treeview(frame1, style='info.Treeview')
displayFiles1.heading("#0", text = "File Path", anchor='w')           # define heading for file path
displayFiles1.place(relwidth=1, relheight=1)

# button: remove files
removeButton = ttk.Button(root, text = 'Delete File', style='secondary.Outline.TButton', command=removeFile)
removeButton.place(relx = 0.48, rely = 0.75, relwidth=0.15)

# button: analyze results
analyzeButton = ttk.Button(root, text = 'Analyze Results', style='info.TButton',  command=openNewWindow)
analyzeButton.place(relx = 0.65, rely = 0.75, relwidth=0.15)

# button: close app
closeButton = ttk.Button(root, text = 'Close Application', style='info.TButton',  command=root.destroy)
closeButton.place(relx=0.5, rely=0.85, anchor='n')

# event loop
root.mainloop()