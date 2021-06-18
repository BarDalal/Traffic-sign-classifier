# -*- coding: utf-8 -*-

"""
Name: Bar Dalal

Traffic Sign Classifier
"""


# Import the necessary libraries:
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
from termcolor import colored
from dataset import Dataset
from model import Model


# Constants:
ORIGINAL_PATH = r'D:\dataset' # the path to the folder that contains the dataset
FILE = r'D:\anaconda\projects\classes_names.txt' # the path to the txt file that contains the names of the classes
                                                # each line in the file is a name of a class
ICON = r'D:\anaconda\projects\ferrari.ico' # the path to the windows' icon
BACKGROUND1 = r'D:\anaconda\projects\background1.jpg' # the path to the first window's background
BACKGROUND2 = r'D:\anaconda\projects\background2.png' # the path to the first window's background
PROGRAMMER_INFO = r'D:\anaconda\projects\programmer.txt' # the path to the file that contains information on the programmer
PROJECT_INFO = r'D:\anaconda\projects\project.txt' # the path to the file that contains information on the project
SAVE = r'D:\anaconda\projects' # the path to the folder in which files the code creates are saved


def get_from_user(path, path_box):
    """
    Parameters
    ----------
    path : A place holder for string variabels, specifically for the path which is gotten from the user.
    path_box : The widget of the entry box.

    If the given path doesn't exist, the function sets it in the placeholder and closes the GUI window.
    Otherwise, the function displays an error message.
    """
    correct = False # True if the given directory is correct, false otherwise
    while not correct:
        if path_box.get() == '': # input is empty
            messagebox.showerror(title = "Error", message = "Please type a path")
            break
        elif os.path.exists(path_box.get()): # the input path already exists
            messagebox.showerror(title = "Error", message = "The path already exists")
            break
        else:
            path.set(path_box.get()) # set the input in the placeholder
            correct = True
            root.destroy() # close the window


def show_info(flag):
    """
    Parameters
    ----------
    flag : True for presenting the information on the programmer and False for the one on the project.

    The function prints information from a txt file according to the given flag.
    """
    if flag:
        with open(PROGRAMMER_INFO) as f:
            info = f.read()
            print(colored("Here's some information on the programmer:", 'red'))
    else:
        with open(PROJECT_INFO) as f:
            info = f.read()
            print(colored("Here's some information on the project:", 'red'))
    print(info)  
    

def no_weights():
    """
    The function displays an error message, because there aren't saved weights.
    """
    pop = Tk()
    pop.wm_withdraw()
    messagebox.showerror("Error", "No weights to load the model with")
    pop.destroy()
    

def check_acc(model, path, size, classes):
    """
    Parameters
    ----------
    model: A keras model.
    path : A string which is a path to the organized dataset.
    size: A integer which is the edge size of the input images.
    classes : A list that contains the names of the different classes.
    
    The function prints the success percentage of the model in predicting all the images of each class.
    """
    model.load_weights(os.path.join(SAVE, 'my_weights.h5')) # load the weights to the neural network
    test_path = os.path.join(path, "test")
    for class_name in os.listdir(test_path):
        sub_path = os.path.join(test_path, class_name)
        test_imgs = [] # the images for model testing, shown as matrices of pixels
        good = 0 # the number of images which the model guessed correctly
        for file in os.listdir(sub_path):
            img = cv2.imread(os.path.join(sub_path, file))
            img = cv2.resize(img, (size, size)) # the image is a matrix at the right shape for test
            test_imgs.append(img) 
        test_imgs = np.array(test_imgs)
        result = model.predict(test_imgs) # an array including arrays of predictions
        for prediction in result:
            pred_class = predicted_class(prediction, classes)
            if pred_class == class_name:
                good += 1
        success = good / len(test_imgs) * 100 # success percentage
        message = "There is " + str(success) + "% success in predicting " + class_name
        print(colored(message, 'blue'))


def predict(model, img, size, classes):
    """
    Parameters
    ----------
    model : A keras model.
    img : A matrix which represents an image needed to be classified by the model.
    size : An integer which is the edge size of the input images.
    classes : A list that contains the names of the different classes.
    
    Returns
    -------
    pred_class : The name of the class which the model predicted.
    """
    img= cv2.resize(img, (size, size))
    img= np.expand_dims(img, axis = 0)
    model.load_weights(os.path.join(SAVE, 'my_weights.h5'))
    prediction = model.predict(img)
    pred_class = predicted_class(prediction[0], classes)
    return pred_class
    
    
def predicted_class(prediction, classes):
    """
    Parameters
    ----------
    prediction : an array which includes probabilities for each class
    classes : A list that contains the names of the different classes.

    Returns
    -------
    The function returns the model's prediction, namely the name of the class recognized by the model.
    """
    probability = np.amax(prediction) # the highest probability
    pred_class = np.where(prediction == probability)
    arr_index = pred_class[0] # array in which the index of the highest probability
    return classes[arr_index[0]]
    

def result(model, size, image_path, classes):
    """
    Parameters
    ----------
    model: A keras model.
    image_path : A string which is a path to the image that needs to be predicted.
    classes : A list that contains the names of the different classes.
    
    The function displays and shows a message about the predicted class.
    """
    img= cv2.imread(image_path)
    pred_class = predict(model, img, size, classes)
    message= "The predicted class is: " + pred_class
    print(colored(message, 'green'))
    messagebox.showinfo(title = "Prediction", message = message)


def upload(top, model, size, lb, classes):
    """
    Parameters
    ----------
    top : A tkinter window.
    model : A keras model.
    size : An integer which is the edge size of the input images.
    lb : A label, tkinter widget where text or images can be placed.
    classes : A list that contains the names of the different classes.

    The function handles the upload procedure, sets the classify button and shows the predicted class when clicking.
    """
    try:
        file_path = filedialog.askopenfilename() # the path to the chosen file
        uploaded = Image.open(file_path)
        im = ImageTk.PhotoImage(uploaded)
        lb.configure(image = im)
        lb.image = im
        classify_b = Button(top, text = "Classify image", command = lambda: result(model, size, file_path, classes), padx = 10, pady = 5)
        classify_b.configure(bg = '#364156', fg = 'white', font = ('arial', 10, 'bold'))
        classify_b.place(relx = 0.79, rely = 0.46)
    except:
        messagebox.showerror(title = "Error", message = "Please choose an image to classify")


def test(model, path, size, classes):
    """
    Parameters
    ----------
    model : A keras model.
    path : A string which is the path to the organized dataset.
    size : An integer which is the edge size of the input images.
    classes : A list that contains the names of the different classes.

    The function runs the test section.
    """
    if not os.path.isfile(os.path.join(SAVE, 'my_weights.h5')):
        no_weights()
    else:
        check_acc(model, path, size, classes)
        top = Toplevel(win)
        top.geometry("800x600")
        top.title("Traffic Sign Classification")
        top.configure(bg = '#CDCDCD')
        top.resizable(width = False, height = False) # the window size can't be changed
        heading = Label(top, text = "Know Your Traffic Sign", pady = 20, font = ('arial',20,'bold'))
        heading.configure(bg= '#CDCDCD', fg= '#364156')
        heading.pack()
        lb = Label(top)
        b1 = Button(top, text= "upload", command = lambda: upload(top, model, size, lb, classes), padx = 10, pady = 5)
        b1.configure(bg = '#364156', fg = 'white', font = ('arial', 10, 'bold'))
        lb.pack()
        b1.pack(side= BOTTOM, expand= True)
    

def train(model, dataset):
    """
    Parameters
    ----------
    model : A keras model.
    dataset : An instance of the Dataset class.
    
    The function runs the model training and also displays and saves graphs of accuracy and loss.
    """
    dataset.resize(size)
    dataset.name_to_num()
    imgs, labels = dataset.get_imgs_labels()
    imgs = np.array(imgs)
    labels = np.array(labels)
    labels = to_categorical(labels) # num of rows is as the num of labels and num of columns is as the num of classes
    # each line contains binary values- one in the column with the number of the class and the rest are zeroes
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(imgs, labels, epochs = 7, batch_size = 64, validation_split = 0.1)
    model.save_weights(os.path.join(SAVE, 'my_weights.h5'))
    summarize_history(history)
   

def summarize_history(history):
    """
    Parameters
    ----------
    history : History object which records training metrics for each epoch.
        
    The function creates and saves graphs of accuracy and loss.
    """
    # Summarize history for accuracy:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.savefig(os.path.join(SAVE, 'accuracy'))
    plt.show()
    # Summarize history for loss:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.savefig(os.path.join(SAVE, 'loss'))
    plt.show()


def real_time(model, size, classes):
    """
    Parameters
    ----------
    model : A keras model.
    size : An integer which is the edge size of the input images.
    classes : A list that contains the names of the different classes.
    
    The function handles the real time classification.
    """
    if not os.path.isfile(os.path.join(SAVE, 'my_weights.h5')):
        no_weights()
    else:
        cap = cv2.VideoCapture(0) # define a video capture object
        while True:
            # Capture frame by frame
            ret, frame = cap.read()
            cv2.putText(frame, "The predicted class is:", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            pred_class = predict(model, frame, size, classes)
            cv2.putText(frame, pred_class, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            cv2.imshow('Real Time Classification', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # check that the keyboard's language is English
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    

# The main function: change to def main()

# Creating Dataset instance:
dataset = Dataset(ORIGINAL_PATH)
dataset.extract_classes(FILE) 
classes = dataset.get_classes() # the names of the classes

# First window:
root = Tk()
root.geometry('800x600')
root.title('Traffic Sign Classification')
root.resizable(width = False, height = False) # the window size can't be changed
root.iconbitmap(ICON) # it must be an ico file
root.configure(bg = '#CDCDCD')

# Set a menu:
menubar = Menu()
root.config(menu = menubar)
about_menu = Menu(menubar, tearoff = 0)
about_menu.add_command(label = "The programmer", command = lambda: show_info(True))
about_menu.add_command(label = "The project", command = lambda: show_info(False))
menubar.add_cascade(label = "About", menu = about_menu)
exit_menu = Menu(menubar, tearoff = 0)
exit_menu.add_command(label = "Exit", command = root.destroy)
menubar.add_cascade(label = "Exit", menu = exit_menu)

# Set background:
bg_label = Label(root)
bg = Image.open(BACKGROUND1)
bg_im = ImageTk.PhotoImage(bg)
bg_label.configure(image = bg_im)
bg_label.image = bg_im
bg_label.place(x = 0, y = 0)

welcome = Label(root, text = 'WELCOME!', font = ('arial', 64), bg = 'red')
msg = "Type the path you want to create, in which the dataset will be organized:"
msg_label = Label(root, text = msg, font = ('arial', 16), bg = 'gold')
path = StringVar() # a placeholder for the path
path_box = Entry(root, width = 20)
submit = Button(root, text = "submit", command = lambda: get_from_user(path, path_box), bg = 'green', font = ('arial', 12, 'bold'))
b1 = Button(root, text = "class distribution", pady = 20, command = lambda: dataset.class_distribution(SAVE), bg = 'green', font = ('arial', 18, 'bold'))
b2 = Button(root, text = "show samples", padx = 10, pady = 20, command = lambda: dataset.show_samples(SAVE), bg = 'green', font = ('arial', 18, 'bold'))
welcome.place(x = 170, y = 20)
msg_label.place(x = 40, y = 150)
path_box.place(x = 350, y = 200)
submit.place(x = 380, y = 220)
b1.place(x = 120, y = 420)
b2.place(x = 500, y = 420)
root.mainloop()

path = path.get() # the string the user typed
path = r'D:\classification'
dataset.set_path(path)
#dataset.split_dataset()
dataset.organize_dataset()
size = dataset.input_shape()

# Creating Model instance:
model = Model(Sequential(), (size, size, 3), len(classes))
model.build()
model.display()
model.plot(os.path.join(SAVE, 'model.png'))

model = model.get_model() # contains the model itself and not an instance of the class Model

# Second window:
win = Tk()
win.geometry('800x600')
win.title('Traffic Sign Classification')
win.resizable(width = False, height = False) # the window size can't be changed
win.iconbitmap(ICON) # it must be an ico file


# Set a menu:
menubar = Menu()
win.config(menu = menubar)
about_menu = Menu(menubar, tearoff = 0)
about_menu.add_command(label = "The programmer", command = lambda: show_info(True))
about_menu.add_command(label = "The project", command = lambda: show_info(False))
menubar.add_cascade(label = "About", menu = about_menu)
exit_menu = Menu(menubar, tearoff = 0)
exit_menu.add_command(label = "Exit", command = win.destroy)
menubar.add_cascade(label = "Exit", menu = exit_menu)

# Set background:
bg_label = Label(win)
bg = Image.open(BACKGROUND2)
bg_im = ImageTk.PhotoImage(bg)
bg_label.configure(image = bg_im)
bg_label.image = bg_im
bg_label.place(x = 0, y = 0)

instruction = "What would you like to do? \n You can train the model now or test with saved weights"
l= Label(win, text = instruction, padx = 30, pady = 30, bg = 'yellow', fg = 'blue', font = ('arial', 18))
b_train= Button(win, text = 'train', padx = 50, pady = 20, command = lambda: train(model, dataset), bg = '#364156', fg = 'white', font = ('arial', 15, 'bold'))
b_test= Button(win, text = 'test', padx = 50, pady = 20, command = lambda: test(model, path, size, classes), bg = '#364156', fg = 'white', font = ('arial', 15, 'bold'))
b_realtime= Button(win, text = 'real time classification', padx = 30, pady = 20, command = lambda: real_time(model, size, classes), bg = '#364156', fg = 'white', font = ('arial', 15, 'bold'))
l.place(x = 80, y = 20)
b_train.place(x = 100, y = 250)
b_test.place(x = 500, y = 250)
b_realtime.place(x = 260, y = 450)
win.mainloop()


