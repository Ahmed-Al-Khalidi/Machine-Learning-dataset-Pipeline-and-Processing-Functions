import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



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

####

