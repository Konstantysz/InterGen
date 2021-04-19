import tkinter as tk          
from tkinter import ttk

def callGenerator():
    print("XD")

root = tk.Tk()
root.title("InterGen")
tabControl = ttk.Notebook(root)
  

tab1 = ttk.Frame(tabControl)
tabControl.add(tab1, text ='Generator')
tab2 = ttk.Frame(tabControl)
tabControl.add(tab2, text ='Results')
tabControl.pack(expand = 1, fill ="both")

###
### TAB 1 - GENERATOR
###

# Variables for storing input
image_size = tk.IntVar(tab1, 512)
image_number = tk.IntVar(tab1, 10)
min_freq = tk.IntVar(tab1, 15)
max_freq = tk.IntVar(tab1, 30)
min_angle = tk.DoubleVar(tab1, 0.0)
max_angle = tk.DoubleVar(tab1, 3.141592653589793)

# Gui blocks
h1 = ttk.Label(tab1, text ="Please type your generation settings")
h2 = ttk.Label(tab1, text ="General settings")

l1 = ttk.Label(tab1, text ="Image size:", anchor = 'w', width=40)
e1 = ttk.Entry(tab1, textvariable = image_size)
l2 = ttk.Label(tab1, text ="Number of images to generate:", anchor = 'w', width=40)
e2 = ttk.Entry(tab1, textvariable = image_number)

s1 = ttk.Label(tab1, text ="")
h3 = ttk.Label(tab1, text ="Advanced settings (Recommended to leave default)")

l3 = ttk.Label(tab1, text ="Minimal spatial frequency:", anchor = 'w', width=40)
e3 = ttk.Entry(tab1, textvariable = min_freq)
l4 = ttk.Label(tab1, text ="Maximal spatial frequency:", anchor = 'w', width=40)
e4 = ttk.Entry(tab1, textvariable = max_freq)


l5 = ttk.Label(tab1, text ="Minimal orientation angle:", anchor = 'w', width=40)
e5 = ttk.Entry(tab1, textvariable = min_angle)
l6 = ttk.Label(tab1, text ="Maximal orientation angle:", anchor = 'w', width=40)
e6 = ttk.Entry(tab1, textvariable = max_angle)

b1 = ttk.Button(tab1, text="Generate", command=callGenerator)
progress = ttk.Progressbar(tab1, orient = tk.HORIZONTAL, length = 400, mode = 'determinate')

# Grid positioning of blocks
h1.grid(column = 0, row = 0, columnspan = 2, padx = 5, pady = 5)
h2.grid(column = 0, row = 1, columnspan = 2, padx = 5, pady = 5)
l1.grid(column = 0, row = 2)
e1.grid(column = 1, row = 2)
l2.grid(column = 0, row = 3, padx = 5, pady = 5)
e2.grid(column = 1, row = 3, padx = 5, pady = 5)
s1.grid(column = 0, row = 4, padx = 5, pady = 5, columnspan = 2)
h3.grid(column = 0, row = 5, padx = 5, pady = 5, columnspan = 2)
l3.grid(column = 0, row = 6,padx = 5, pady = 5)
e3.grid(column = 1, row = 6, padx = 5, pady = 5)
l4.grid(column = 0, row = 7,padx = 5, pady = 5)
e4.grid(column = 1, row = 7, padx = 5, pady = 5)
l5.grid(column = 0, row = 8,padx = 5, pady = 5)
e5.grid(column = 1, row = 8, padx = 5, pady = 5)
l6.grid(column = 0, row = 9, padx = 5, pady = 5)
e6.grid(column = 1, row = 9, padx = 5, pady = 5)
b1.grid(column = 0, row = 10, padx = 5, pady = 5, columnspan = 2)
progress.grid(column = 0, row = 11, padx = 5, pady = 5, columnspan = 2)

###
### TAB 2 - RESULTS
###
l7 = ttk.Label(tab2, text ="Results", anchor = 'w', width=40).grid(column = 0,row = 0, padx = 5, pady = 5, columnspan = 5)
sb = ttk.Scrollbar(tab2).grid(column = 0,row = 1, padx = 5, pady = 5, columnspan = 2)


root.mainloop()