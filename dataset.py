# -*- coding: utf-8 -*-

# Name: Bar Dalal


import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
import random
import cv2


class Dataset:
    # The class is responsible for organizing the dataset
     
    
    def __init__(self, original_path):
        """
        Parameters
        ----------
        original_path : A string which is a path to the floder that contains the dataset.

        The function creates an instance of the class.
        """
        self.__original_path = original_path # the path to the folder that contains the dataset
        self.__classes = [] # list that contains the names of all classes
        self.__path = "" # the path in which the dataset will be organized
        self.__imgs = [] # a list which contains the images for training that are shown as matrices of pixels
        self.__labels = [] # a list which contains the labels of the images
    
    
    def extract_classes(self, file):
        """
        Parameters
        ----------
        file : A string which is the path to the txt file that contains the names of the classes.
        
        The function gets the classes names from a txt file and updates the classes attribute.
        """
        with open(file) as f: # the file is closed automatically after the block
            classes = f.read().splitlines()
        self.__classes = classes
            

    def class_distribution(self, save):
        """
        Parameters
        ----------
        save: A string which is the path to the folder in which the plot is saved.   
            
        The function displays and saves a bar plot which shows the number of images for each class.
        """ 
        quantities = [] # the number of images in the dataset for each class
        for class_name in self.__classes:
            with ZipFile(os.path.join(self.__original_path, class_name + '.zip')) as zip:
                files = zip.namelist() # the images' names of a certain class 
            quantities.append(len(files))
        plt.figure(figsize = (30, 10))
        plt.bar(self.__classes, quantities)
        plt.title("class distribution of the dataset")
        plt.xlabel('classes')
        plt.ylabel('images')
        plt.savefig(os.path.join(save, 'class distribution.png'))
        plt.show()


    def show_samples(self, save):
        """
        Parameters
        ----------
        save: A string which is the path to the folder in which the samples are saved.
    
        The function displays and saves a random image from each class.
        It extracts one image at a time to the folder it created, and saves the plot.
        The directory created by the function is deleted after all the plots are saved.
        """
        folder = os.path.join(save, 'folder') # the folder to which the samples are extracted
        os.mkdir(folder)
        for class_name in self.__classes:
            with ZipFile(os.path.join(self.__original_path, class_name + '.zip')) as zip:
                files = zip.namelist() # the images' names of a certain class
                img = random.choice(files)
                zip.extract(img, folder)
            img = os.path.join(folder, img) # the path to the image that needs to be shown
            mat_img = cv2.imread(img)
            os.remove(img)  # delete the image
            plt.imshow(cv2.cvtColor(mat_img, cv2.COLOR_BGR2RGB)) # converting from BGR to RGB
            plt.title(class_name)
            plt.savefig(os.path.join(save, class_name))
            plt.show()
        os.rmdir(folder) # delete the folder that was for extracting
        
    
    def split_dataset(self):
        """
        The function creates the organized dataset, that is divided to train (70%) and test (30%) randomly.
        It extracts all the images from zip files.
        """
        # creating the folders:
        os.mkdir(self.__path)
        os.mkdir(os.path.join(self.__path, "train"))
        os.mkdir(os.path.join(self.__path, "test"))
        for class_name in self.__classes:
            os.mkdir(os.path.join(self.__path, "train", class_name))
            os.mkdir(os.path.join(self.__path, "test", class_name))
            # tranferring the images:
            sub_path = os.path.join(self.__original_path, class_name + '.zip') # path to zip file of a certain class
            with ZipFile(sub_path) as zip:
                num = int(0.3 * len(zip.namelist())) # how many it's 30% of the images of this class
                zip.extractall(os.path.join(self.__path, 'train', class_name))
            transfer = random.sample(os.listdir(os.path.join(self.__path, 'train', class_name)), num) # list of the images to transfer, chosen randomly
            for img in transfer:
                os.rename(os.path.join(self.__path, 'train', class_name, img), os.path.join(self.__path, 'test', class_name, img))
    
    
    def organize_dataset(self):
        """
        The function updates the imgs and labels attributes of the class so:        
            imgs : a list which contains the images for training that are shown as matrixes of pixels.
            labels : a list which contains the names of the objects that is in the images.
                     The name in a certain index belongs to the image with the same index in the imgs list.
        """
        train_path = os.path.join(self.__path, "train") # the path for the folder which contains the images for the training
        for class_name in os.listdir(train_path): # pass over the images of the different categories
                for img in os.listdir(os.path.join(train_path, class_name)):
                    self.__imgs.append(cv2.imread(os.path.join(train_path, class_name, img)))
                    self.__labels.append(class_name)
    
    
    def input_shape(self):
        """
        Returns
        -------
        size : the average size of the edges of the images for training in the dataset.
        """
        heights = [] # a list which contains the heights of all images
        widths = [] # a list which contains the widths of all images
        for img in self.__imgs:
            heights.append(img.shape[0])
            widths.append(img.shape[1])
        heights_average = int(sum(heights) / len(heights))
        widths_average = int(sum(widths) / len(widths))
        size = int((heights_average + widths_average) / 2)
        return size


    def resize(self, size):
        """
        Parameters
        ----------
        size : an integer which is the desirable size of the input image.
        
        The function resizes all the images according to the given size.
        """
        for i, img in enumerate(self.__imgs):
            self.__imgs[i] = cv2.resize(img, (size, size))
        

    def name_to_num(self):
        """  
        The function updates the labels attribute so it's a list of integers.
        The function converts the classes names to numbers, so each integer number represents a class.
        """
        converter = {} # the keys are the classes names and the values are the numbers which associated with them
        num = 0 # the class number
        for name in self.__classes:
            converter[name] = num
            num += 1
        for i, name in enumerate(self.__labels): # i is the index and name is the value within this index
            self.__labels[i] = converter[name] # convert the class name to a number

    
    def get_classes(self):
        """
        Returns
        -------
        The list which contains the names of the classes.
        """
        return self.__classes

    
    def get_imgs_labels(self):
        """
        Returns
        -------
        The function returns two lists: the first is the value of imgs attribute 
        and the second is the value of labels attribute.
        """
        return self.__imgs, self.__labels
    
    
    def set_path(self, path):
        """
        Parameters
        ----------
        path : A string which is a path to the floder that contains the organized dataset.

        The function updates the value of the path attribute.
        """
        self.__path = path
        



