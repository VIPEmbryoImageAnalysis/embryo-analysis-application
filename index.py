import os
import tkinter as tk
import tkinter.ttk as ttk

from tkinter import filedialog
from tkinter.ttk import Combobox
from tkinter import messagebox

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
    
    yesRemoveButton = tk.Button(newWindow, width = 5, text = 'Yes') # yes button
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
        label = tk.Label(newWindow, text = "Please wait...")
        label.pack()

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

