from tkinter import *
from tkinter.ttk import Progressbar
import tkinter.messagebox
from tkinter.ttk import Progressbar, Style, Button
import cv2, os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import ImageTk, Image
from inference import SurvedModel
mpl.rcParams['figure.dpi'] = 300
import time
from threading import Thread

##################################################
# Declair variables and create class instances
##################################################


# For details regarding paprika in drop down menu
details = 'BraTS is acronym of Multimodal Brain Tumor Segmentation. The details of dataset can be found at \
            \n https://www.med.upenn.edu/cbica/brats2020/data.html'
            
help_msg = 'An implementation of paper named.\n"BU-Net: Brain Tumor Segmentation Using Modified U-Net Architecture" \nby Talha Ilyas'

##################################################
# Decair functions you will use in your GUI
##################################################
def model_loader():
    '''For loading the model on GPU by givin dummy input'''
    global  model
    model = SurvedModel()
    #_ = model.predict(np.ones((600,600,3)).astype(np.uint8))
    
def paprika():
    '''For Details regarding program in drop down menue'''
    tkinter.messagebox.showinfo(title='BraTS', message=details)

def fct():
    '''Model loading progress bar'''
    for i in range(1, 101):
        # because it takes about ~30 seconds for the model to be uploaded on GPU => 0.3*100
        time.sleep(0.01)
        progress.step()
        if i != 100:
            s.configure("LabeledProgressbar", text="Loading Model on GPU please wait: {0} %      ".format(i))
        elif i == 100:
            s.configure("LabeledProgressbar", text="Done Loading Model")
        root.update()
                               
def bar(): 
    '''For makin undeterminstic progress bar (not used in this script)'''
    steps = [0, 20, 40, 50, 60, 80, 100, 80, 60, 50, 40, 20]
    for i in steps:
        progress_det['value'] = i
        root.update_idletasks() 
        time.sleep(0.00009)
    progress_det['value'] = 0

def increment():
    '''The detection progress bar'''
    for i in range(100):
        progress_det["value"] = i+1
        root.update()
        time.sleep(0.00009)
        
def about_us():
    '''About us message in drop down menue'''
    tkinter.messagebox.showinfo(title='Robot Vision Lab', message = help_msg)
    
def browse_file():
    '''For loadinf file in model via drop down menue'''
    # to clear frame
    for widget in resultframe.winfo_children():
        widget.destroy()
    global img_path, img, name
    
    img_path = filedialog.askopenfilename()
    img = np.load(img_path)
    img2 = (img[:,:, 0] * 255).astype(np.uint8)
    img2 = cv2.resize(img2, (640,640))# just so it can fit inside root window
    img2 = cv2.merge((img2,img2,img2))
    
    name = os.path.basename(img_path)
    photo2 = ImageTk.PhotoImage(Image.fromarray(img2)) 
    
    # label widget also acts as a conteiner so we can use it to embed image
    labelphoto2 = Label(resultframe, image = photo2)
    labelphoto2.img2 = photo2
    labelphoto2.pack(side=LEFT, padx=5)
    statusbar['bg'] = 'white'
    statusbar['text'] = 'Loaded Scan: {}'.format(name)

def play_btn():
    '''Command function for when you click "Detect" button '''
    try:
        print('Running')
        op = model.predict(img)
        print('Done')
        increment()
        op = cv2.resize(op, (640,640))
        
        photo3 = ImageTk.PhotoImage(Image.fromarray(op))
        labelphoto3 = Label(resultframe, image = photo3)
        labelphoto3.img = photo3
        labelphoto3.pack(side=RIGHT, padx=5)
        
        statusbar['bg'] = 'green'
        statusbar['text'] = 'Done'
    except NameError:
        tkinter.messagebox.showerror(title='File not found', message='Load BraTS scan before running the segmentation model')

##################################################
# Now start creating window GUI
##################################################
PATH    = os.getcwd()
# creat a window and store in inside root variable it'll be created for 
# milisecondes 
root = Tk()
# frame 1 for title
titleframe = Frame(root)
titleframe.pack(padx=10, pady=10)
# isolation frame 1 for ip/op images
resultframe = Frame(root, relief=RAISED, borderwidth=1)# , relief=RAISED, borderwidth=1
resultframe.pack(padx=10, pady=10)
# command frame for detect button and progress bars
commandframe = Frame(root)
commandframe.pack(padx=10, pady=10)

# add title of main root window
root.title('Disease Detector')
# add icon of main root window
#root.iconbitmap(r'C:/Users/Talha/Desktop/chkpt/paprika_model/icon.ico')
# increase the size of window so that when script is run it opens a 1500x1500px window
root.geometry('1500x1500')

#*****************************
# creat a menubar
menubar = Menu(root)
root.config(menu = menubar)

# creat submenu
submenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(labe='File', menu=submenu)
submenu.add_command(label='Load Scan', command= browse_file)
submenu.add_command(label='About BraTS', command=paprika)
# creat submenu 2
submenu2 = Menu(menubar, tearoff = 0)
menubar.add_cascade(labe='Help', menu=submenu2)
submenu2.add_command(label='About Us', command = about_us)
submenu2.add_command(label='Exit', command=root.destroy)

#*******************************
# Header Line of the GUI
# add photo
photo = PhotoImage(file= PATH + '/rv.png')
# label widget also acts as a conteiner so we can use it to embed image
labelphoto = Label(titleframe, image = photo)
labelphoto.pack(side=LEFT)
# Label widget 
text = Label(titleframe, text='Load Brain Scans for Tumor Segmentation.', fg = "black",font = "Helvetica 16 bold italic")# bg = "white"
# now you'll have to pack it inside tkinter window
text.pack(side=LEFT, padx = 10)

#*************************************************************************************************************************
# Initilizing progress bars

# progressbar with text inside it
s = Style(root)
# add the label to the progressbar style
s.layout("LabeledProgressbar",
         [('LabeledProgressbar.trough',
           {'children': [('LabeledProgressbar.pbar',
                          {'side': 'left', 'sticky': 'ns'}),
                         ("LabeledProgressbar.label",   # label inside the bar
                          {"sticky": ""})],
           'sticky': 'nswe'})])

progress = Progressbar(commandframe, orient="horizontal", length=300, style="LabeledProgressbar")
progress.pack(side=TOP, padx=10, pady=10)
# change the text of the progressbar, 
# the trailing spaces are here to properly center the text
s.configure("LabeledProgressbar", text="Loading Model on GPU please wait:0 %      ")

# uncomment this if you want to make an image button instead of 'Detect' text button
#btnphoto = ImageTk.PhotoImage(Image.open('C:/Users/Talha/Desktop/chkpt/paprika_model/run.png'))
# make detect button
btn = Button(commandframe, text='Detect',  command =play_btn)#image = btnphoto,
btn.pack(side=LEFT, padx=10, pady=10)

# 2nd progress bar with detect button
progress_det = Progressbar(commandframe, length=200, cursor='watch',mode="determinate", orient=HORIZONTAL)
#progress_det = Progressbar(commandframe, orient = HORIZONTAL, length = 100, mode = 'indeterminate') # for shuttling block progress bar
progress_det.pack(side=LEFT, padx=10, pady=10)

##################################################
# Start threadin b/c loading model on GPU will take time
# threading will run processes in parallel so that our GUI don't stop responding
##################################################
# Start loading model bar
thread1 = Thread(target=fct, daemon=True)
thread1.start()
print('thread 1 start')

# Now start loading mdoel on GPU in parallel
thread2 = Thread(target=model_loader, daemon=True)
thread2.start()
print('thread 2 start')

# make a Bottom statusbar for knowing the status of program
statusbar = Label(root, text = 'Welcome to Brain Tumor Segmenter', relief = SUNKEN, anchor=W)
statusbar.pack(side=BOTTOM, fill=X)

# it'll run the loop in an infinite loop (if you comment it it'll might give some error)
root.mainloop()
