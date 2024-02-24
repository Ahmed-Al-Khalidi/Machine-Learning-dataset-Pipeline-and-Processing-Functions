import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Function 1 images pipeline loader

def image_pipeline(folder_path):
    images_data = []
    labels = []
    Classes = os.listdir(folder_path)

    for i in range(len(Classes)):
        new_path = folder_path + '/' + Classes[i]
        Classes2 = os.listdir(new_path)

        for j in range(len(Classes2)):
            labels.append(Classes[i])

            image_path = folder_path + '/' + Classes[i] + '/' + Classes2[j]
            
            try:
                # Attempt to open the image
                image = Image.open(image_path)
                
                # Convert the image to a NumPy array
                image_array = np.array(image)
                
                # Append the image array to the list
                images_data.append(image_array)
            
            except (IOError, OSError) as e:
                # Handle the exception (e.g., print a warning, skip the image, etc.)
                print(f"Error processing {image_path}: {e}")

    return images_data, labels



# Function 2 Divide the dataset to train, validation and test datasets

def Dataset_division(features,labels,val_per,test_per):
    # Create lists for training, validation data and lists for test data
    train_features=[]
    train_labels=[]
    #################
    val_features=[]
    val_labels=[]
    ################
    test_features=[]
    test_labels=[]
    # Calculate the number of validation data and test data from the original data
    val_len=int(val_per*len(labels))
    test_len=int(test_per*len(labels))
    train_len=((len(labels)-(val_len+test_len)))
    # Start dividing the data
    for i1 in range(train_len):
        train_features.append(features[i1])
        train_labels.append(labels[i1])

    for i2 in range(val_len):
        val_features.append(features[(i2+train_len)])
        val_labels.append(labels[i2+train_len])

    for i3 in range(test_len):
        test_features.append(features[i3+train_len+val_len])
        test_labels.append(labels[i3+train_len+val_len])

    return train_features,train_labels,val_features,val_labels,test_features,test_labels



# Function 3 image rescalling the pixels to be between 0 to 1 

def Image_scalling(data_list):
    list_length=len(data_list)
    new_data_list=[]
    for i in range(list_length):
        new_data_list.append((data_list[i]/255))

    return new_data_list



# Function 4 RGB to grayscale

def RGB_to_grayscale(image_array):
    length= len(image_array)
    images_list=[]
    for i in range(length):
        # Compute the weighted sum to convert to grayscale
        images_list.append(np.dot(image_array[i], [0.299, 0.587, 0.114]))

    return images_list



# Function 5 Pipeline with image resizing feature

def image_pipeline_with_resize(folder_path,height,width):
    images_data = []
    labels = []
    size=(height,width)
    Classes = os.listdir(folder_path)

    for i in range(len(Classes)):
        new_path = folder_path + '/' + Classes[i]
        Classes2 = os.listdir(new_path)

        for j in range(len(Classes2)):
            labels.append(Classes[i])

            image_path = folder_path + '/' + Classes[i] + '/' + Classes2[j]
            
            try:
                # Attempt to open the image
                image = Image.open(image_path)

                # Resize the images
                img_resized = image.resize(size, Image.Resampling.LANCZOS)
                
                # Convert the image to a NumPy array
                image_array = np.array(img_resized)
                
                # Append the image array to the list
                images_data.append(image_array)
            
            except (IOError, OSError) as e:
                # Handle the exception (e.g., print a warning, skip the image, etc.)
                print(f"Error processing {image_path}: {e}")

    return images_data, labels



