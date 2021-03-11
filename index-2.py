import os
import tkinter as tk
import tkinter.ttk as ttk
import results
from tkinter import *

from tkinter import filedialog
from tkinter.ttk import Combobox
from tkinter import messagebox
from results import *
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# open new window
root = tk.Tk()
root.title("Embryo Analysis")
root.geometry("900x600")

# define event: open file explorer, select video, display selected video in table
def uploadVideos():
    filepath = filedialog.askopenfilename()
    displayFiles.insert('', 'end', text= filepath, values=(filepath))

# define event: confirm removal of file
def confirmFileRemoval():
    newWindow = tk.Toplevel(root)
    newWindow.title("Delete File")
    newWindow.geometry("300x200")
    ttk.Label(newWindow, text = "Are you sure you want to remove the \n\t   selected video?", font = ("Arial", 10)).place(x = 40, y = 30)     # confirmation message
    
    yesRemoveButton = tk.Button(newWindow, width = 5, text = 'Yes', command = removeFile) # yes button
    yesRemoveButton.place(x = 100, y = 120)
    
    cancelRemoveButton = tk.Button(newWindow, text = 'Cancel', command = root.destroy)  # cancel button
    cancelRemoveButton.place(x = 160, y = 120)


# define event: remove selected file
def removeFile(): 
    try:
        selected_item = displayFiles.selection()[0] # get selected item
        displayFiles.delete(selected_item)
    except IndexError:
        messagebox.showerror("Error", "No selected file") # error message


# define event: open new window
def openNewWindow():
    try:
        newWindow = tk.Toplevel(root)
        newWindow.title("Analyze Results")
        newWindow.geometry("1300x600")
        label = tk.Label(newWindow, text = "Pleasee wait...")
        sizeResults_tree = ttk.Treeview(newWindow)
        label.pack()

        data1 = {'Embryo_Size': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010],
         'Time': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
        }

        data2 = {'Embryo_Size': [1940,1950,1960,1970,1980,1990,2000,2010,2011,2012],
         'Time': [19.8,11,3,5.2,2.9,10,8.5,9.2,9.5,8.3]
        }
        

        df1 = DataFrame(data1,columns=['Embryo_Size','Time'])
        df2 = DataFrame(data2,columns=['Embryo_Size','Time'])

        frame = tk.Frame(newWindow)
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111)
        line2 = FigureCanvasTkAgg(figure2, newWindow)
        line2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        df1 = df1[['Embryo_Size','Time']].groupby('Embryo_Size').sum()
        df1.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=5)        
        df2 = df2[['Embryo_Size','Time']].groupby('Embryo_Size').sum()
        df2.plot(kind='line', legend=True, ax=ax2, color='b',marker='o', fontsize=5)
        ax2.set_title('Embryo Growth Over Time')

        frame = tk.Frame(newWindow ,borderwidth=5, relief="groove" )
        frame.place(x=700, y=5000)
#         frame = Frame(W, width=100, height=50)
        # frame.place(x=700, y=0)
        label = tk.Label(frame, text="test", borderwidth=5, relief="groove" )

        frame.pack()
        col1=[['Video','Rank'],['testOne', 1],['testTwo', 2],['testThree', 3]]
        col2=['Video','Rank']
        for r in range(len(col1)):            
            for c in range(len(col2)):
                tk.Label(frame, text='%s'%(col1[r][c]),
                    borderwidth=5 ).grid(row=r,column=c)


        frame = tk.Frame(newWindow, borderwidth=5, relief="groove" )
        frame.place(x=700, y=700)
        frame.pack()
        col1=[['Video','Rank'],['testOne', 1],['testTwo', 2],['testThree', 3]]
        col2=['Video','Rank']
        for r in range(len(col1)):            
            for c in range(len(col2)):
                tk.Label(frame, text='%s'%(col1[r][c]),
                    borderwidth=5 ).grid(row=r,column=c)   

        frame = tk.Frame(newWindow, borderwidth=5, relief="groove" )
        frame.pack()
        col1=[['Video','Final Size', 'Average Growth Rate'],['testOne', 27722.724, 458.516]]
        col2=['Video','Final Size', 'Average Growth Rate']
        for r in range(len(col1)):            
            for c in range(len(col2)):
                tk.Label(frame, text='%s'%(col1[r][c]),
                    borderwidth=1 ).grid(row=r,column=c)       

        frame = tk.Frame(newWindow, borderwidth=5, relief="groove" )
        frame.pack()

        yesRemoveButton = tk.Button(frame, width = 20, text = 'Save Images and Labels').pack()# yes button
        # yesRemoveButton.place(x = 1000, y = 700)
        yesRemoveButton = tk.Button(frame, width = 20, text = 'Save Results').pack()
        yesRemoveButton = tk.Button(frame, width = 20, text = 'Load new videos').pack()
        yesRemoveButton = tk.Button(frame, width = 20, text = 'Close App').pack()

        # figure2 = plt.Figure(figsize=(5,4), dpi=100)
        # ax2 = figure2.add_subplot(111)
        # line2 = FigureCanvasTkAgg(figure2, newWindow)
        # line2.show()        
        # line2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # frame.pack()

        # x = np.array([1, 2, 3, 4]) 
        # y = x*2
        
        # # first plot with X and Y data 
        # plt.plot(x, y) 
        
        # x1 = [2, 4, 6, 8] 
        # y1 = [3, 5, 7, 9] 
        
        # # second plot with x1 and y1 data 
        # plt.plot(x1, y1, '-.') 
        
        # plt.xlabel("X-axis data") 
        # plt.ylabel("Y-axis data") 
        # plt.title('multiple plots') 
        # plt.show() 

    except IndexError:
        messagebox.showerror("Error", "No uploaded files")  # error message

# button: upload videos
uploadButton = tk.Button(root, text = 'Upload Video', font = ("Arial", 11), command = uploadVideos)
uploadButton.place(x = 400, y = 90)

#display selected files
ttk.Label(root, text = "Uploaded files:", font = ("Arial", 11)).place(x = 100, y = 180)  # label for selected files

displayFiles = ttk.Treeview(root)
# displayFiles["columns"] = ["filepathHeading"]  # define columns: #0 - first column by default

displayFiles.column("#0", width = 500, minwidth = 200, stretch = tk.NO)             # define column style for file name
# displayFiles.column("filepathHeading", width = 300, minwidth = 250, stretch = tk.NO)       # define column style for file path

displayFiles.heading("#0", text = "File Path", anchor = tk.W)           # define heading for file name
# displayFiles.heading("filepathHeading", text = "File Path", anchor = tk.W)     # define heading for file path

displayFiles.place(x = 250, y = 180)

# button: remove files
removeButton = tk.Button(root, text = 'Delete File', command = removeFile)
removeButton.place(x = 590, y = 420)

# button: analyze results
analyzeButton = tk.Button(root, text = 'Analyze Results', command = openNewWindow)
analyzeButton.place(x = 660, y = 420)

# button: close app
closeButton = tk.Button(root, text = 'Close Application', font = ("Arial", 11), command = root.destroy)
closeButton.place(x = 400, y = 500)

# event loop
root.mainloop()

