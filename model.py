# -*- coding: utf-8 -*-

# Name: Bar Dalal


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from tensorflow.keras.utils import plot_model
from termcolor import colored


class Model:
    # The class is responsible for the model, namely the neural network which classifies the road signs
    
    
    def __init__(self, model, input_shape, num_classes):
        """
        Parameters
        ----------
        model : A Keras model, which is intended to classify the different images.
        input_shape : A tuple which is the shape of the images that are input of the model. 
        num_classes : An integer which is the number of the classes to classify to.

        The function creates an instance of the class.
        """
        self.__model = model # the model which classifies the images
        self.__input_shape = input_shape # the shape of an image that pass through the model
        self.__num_classes = num_classes # the number of the classes that the model can classify the images to
        self.__layers = [] # a list which contains the layers of the model
    
    
    def add_layers(self):
        """
        The function updates the layers attribute.
        """
        layers = [] # list of the model layers
        layers.append(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', input_shape = self.__input_shape))
        layers.append(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu'))
        layers.append(MaxPooling2D(pool_size = (2, 2)))
        layers.append(Dropout(rate = 0.25))
        layers.append(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
        layers.append(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
        layers.append(MaxPooling2D(pool_size = (2, 2)))
        layers.append(Dropout(rate = 0.25))
        layers.append(Flatten())
        layers.append(Dense(256, activation = 'relu'))
        layers.append(Dropout(rate = 0.5))
        layers.append(Dense(self.__num_classes, activation = 'softmax'))
        self.__layers = layers    
    
    
    def build(self):
        """
        The function builds the model by updating the layers and model attributes.
        """
        self.add_layers()
        for layer in self.__layers:
            self.__model.add(layer)
    
    
    def display(self):
        """
        The function displays a textual summary of the model.
        """
        print(colored("Here is a summary of the model:", 'red'))
        self.__model.summary()
        
        
    def plot(self, file):
        """
        Parameters
        ----------
        file : A string which is the path to the saved plot image.
        
        The function creates a plot to the model and saves it at the given path.
        """
        plot_model(self.__model, to_file = file, show_shapes = True, show_layer_names = True)
        
        
    def get_model(self):
        """
        Returns
        -------
        The model.
        """
        return self.__model



   
