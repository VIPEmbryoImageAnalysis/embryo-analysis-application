import os
import tkinter as tk
import tkinter.ttk as ttk
import index

from tkinter import filedialog
from tkinter.ttk import Combobox
from tkinter import messagebox

# define event: open new window
def openNewWindow():
    try:
        newWindow = tk.Toplevel(root)
        newWindow.title("Analyze Results")
        newWindow.geometry("1300x600")

        # label: waiting for results
        label = tk.Label(newWindow, text = "Please wait...")
        label.pack()

        # tree: final embryo size
        finalSize_tree = ttk.Treeview(newWindow)
        finalSize_tree["columns"] = ["rankHeading"]  # define columns: #0 - first column by default

        finalSize_tree.column("#0", width = 250, minwidth = 200, stretch = tk.NO)             # define column style for video title
        finalSize_tree.column("rankHeading", width = 200, minwidth = 150, stretch = tk.NO)       # define column style for rank

        finalSize_tree.heading("#0", text = "Video", anchor = tk.W)           # define heading for video title
        finalSize_tree.heading("rankHeading", text = "Rank", anchor = tk.W)     # define heading for rank

        finalSize_tree.place(x = 250, y = 180)

    except IndexError:
        messagebox.showerror("Error", "No uploaded files")  # error message
