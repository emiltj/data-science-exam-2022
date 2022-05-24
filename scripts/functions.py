import cv2, keras
import numpy as np
import pandas as pd
from keras import layers
from keras.utils import np_utils
from keras.datasets import mnist
from scipy.signal import medfilt
from sklearn.preprocessing import normalize
from sklearn import svm, metrics, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Define function for making binary
def make_binary(img, threshold): 
    """Binarizes a 2D image on the basis of a threshold

    Args:
        img (np.ndarray): 2D array to binarize
        threshold (int): Threshold for binarizing upon

    Returns:
        np.ndarray: Binarized image
    """

    img[img >= threshold] = 255
    img[img < threshold] = 0
    
    return img

# Define function for Game of Life
def GoL(seed = np.ndarray, n_generations = int):
    """Performs Game of Life simulations

    Args:
        seed (np.ndarray): Image seed, to perform GoL on  Defaults to np.ndarray.
        n_generations (int): Number of generations to perform. Defaults to int.

    Returns:
        List: List of generations
    """
    
    # Empty list for appending generations to (and start with the seed)
    generations = []
    
    # Append seed to list of generations
    generations.append(seed)
    
    # Apply 1-layer 0-padding
    seed = np.pad(seed, 1)

    # Define n_rows and n_cols from shape of img
    n_rows, n_cols = seed.shape

    # Perform ticks
    for i in range(n_generations):
        # Create image for next step, for overwriting
        generation = np.array(np.zeros(shape=(n_rows, n_cols), dtype=np.int32))
        
        # For loop that iterates over each cell in the array
        for r in range(n_rows-2):
            for c in range(n_cols-2):
                
                # seed[r+1, c+1] (the "middle" cell during each window) and sum_context (sum of all cells around "middle" cell in window) ...
                # ... has the right information. Check with: print(seed[r+1, c+1]) and print(sum_context)
                sum_context = seed[r, c] + seed[r, c+1] + seed[r, c+2] + seed[r+1, c] + seed[r+1, c+2] + seed[r+2, c] + seed[r+2, c+1] + seed[r+2, c+2]

                # Any live cell with fewer than 2 or more than 3, dies
                if seed[r+1, c+1] == 1*255:
                    if sum_context < 2*255 or sum_context > 3*255:
                        generation[r+1, c+1] = 0
                
                # Any live cell with two or three live neighbours lives, unchanged
                if seed[r+1, c+1] == 1*255 and 4*255 > sum_context > 1*255:
                    generation[r+1, c+1] = 1*255

                # Any dead cell with exactly 3 three live neighbours will come to life
                if seed[r+1, c+1] == 0 and sum_context == 3*255:
                    generation[r+1, c+1] = 1*255
        
        # Assign newest generation as the new seed
        seed = generation.copy()

        # Append newest generation to list of generations 
        generations.append(generation[1:-1, 1:-1])
    
    return generations

# Define q(d, l) - Function that calculates impending corrosion speed - also based on y.
def Q(d, l):
    return (255 - d) * l

# Define function for corrosion
def corrosion(seed: np.ndarray, n_generations: int, l: float, v: int, Q):
    """Performs corrosion generations over time, as described in paper by Horsmans, Grøhn & Jessen, 2022

    Args:
        seed (np.ndarray): Image seed, to perform corrosion on - Defaults to np.ndarray.
        n_generations (int): Number of generations to perform. Defaults to int.
        l (float): Number describing how much corrosion takes place
        v (int): Number describing the threshold for "smooth surfaces" (i.e. surfaces where corrosion doesn't happen)
        Q (function): Function that calculates impending corrosion speed - also based on y.
    """    
    # Empty list for appending generations to (and start with the seed)
    generations = []

    # Append seed to list of generations
    generations.append(seed)

    # Apply 1-layer reflective-padding
    seed = np.pad(seed, mode = 'reflect', pad_width = 1)

    # Define n_rows and n_cols from shape of img
    n_rows, n_cols = seed.shape

    for generation in range(n_generations):

        # Create image for next step, for overwriting
        generation = np.array(np.zeros(shape=(n_rows, n_cols), dtype=np.int32))

        for r in range(n_rows-2):
            for c in range(n_cols-2):
                
                # seed[r+1, c+1] (the "middle" cell during each window) and context (cells around "middle" cell in window) ...
                # ... has the right information. Check with: print(seed[r+1, c+1]) and print(context)
                context = [seed[r, c], seed[r, c+1], seed[r, c+2], seed[r+1, c], seed[r+1, c+2], seed[r+2, c], seed[r+2, c+1], seed[r+2, c+2]]

                # d (Difference) is difference between center and the lowest in the context
                d = seed[r+1, c+1] - np.min(context)

                # Any cell with difference > v and difference < 255 changes value to previous_value + q(d, l)
                if 255 >= d and d >= v:
                    generation[r+1, c+1] = seed[r+1, c+1] + Q(d, l)

                # Any cell with difference smaller than v or with difference larger than 255, then the new generation has the same value as large generation
                if d < v or d > 255:
                    generation[r+1, c+1] = seed[r+1, c+1]
            
        # Assign newest generation as the new seed
        seed = generation.copy()

        # Append newest generation to list of generations 
        generations.append(generation[1:-1, 1:-1])
    
    # Return generations
    return(generations)

# Define function for calculating measure of corrosion-increase-from-baseline on an entire feature set
def corrosion_increase(X, y, n_generations, l, v, Q, mean_cells_active):
    """Function for calculating measure of corrosion-increase-from-baseline on an entire feature set

    Args:
        X (np.nd.array): 3D array with dim(samples, 1st_dimension_of_img, 2nd_dimension_of_img)
        y (np.nd.array): 1D array with labels for images
        n_generations (int): Number of generations to perform
        l (float): Number describing how much corrosion takes place
        v (int): Number describing the threshold for "smooth surfaces" (i.e. surfaces where corrosion doesn't happen)
        Q (function): Function that calculates impending corrosion speed - also based on y.
        mean_cells_active (list): List of length 10, with average number of active cells for each label (0, 1, 2, etc.)
    """
    corrosion_increase_by_number = []

    # For 0, 1, 2, ... len(X):
    for i in range(len(list(X))):
        
        # Define seed, sum of seed and class of seed
        seed = X[i]
        sum_seed = sum(seed.flatten())
        class_of_seed = y[i]
        
        # Generations of corrosion
        generations = corrosion(seed, 8, 0.1, 6, Q)

        corrosion_increases = []

        # For each generation, calculate ??? (Jakob, definér?)
        for i in generations:
            sum_generation = sum(i.flatten())
            active_in_img = len(i[i>0])
            
            # (generation_cell - seed_cell) * (avg_active_pixels_for_number / active_pixels_for_current_img)
            corrosion_increases.append((sum_generation - sum_seed)) #* (mean_cells_active[class_of_seed] / active_in_img))
        
        corrosion_increase_by_number.append(corrosion_increases)

    return(corrosion_increase_by_number)

def ML(feature_sets, feature_set_names, y):
    """Function for performing machine learning. Outputs performance metrics

    Args:
        feature_sets (List): List of feature sets. The elements in the list contains 4D arrays of dim = (batch, 1st dim of image, 2nd dim of image, 1)
        feature_set_names (_type_): List of the names of the feature sets
        y (np.ndarray): Array containing true value of image

    Returns:
        dict: Nested dictionary with:
            1st level keys = feature_set_names, 
            2nd level keys = lr, cnn
            3rd level keys = confusion_matrix, classification_report
            
            e.g.:
            dict_keys(['X', 'X_bina', 'X_GoL'])
            dict_keys(['lr', 'cnn'])
            dict_keys(['confusion_matrix', 'classification_report'])

    """
    ############################################# Define models #############################################
    
    # Define CNN model
    epochs = 15
    input_shape = (28, 28, 1)
    num_classes = len(set(y))
    cnn = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Define logistic regression model
    lr = LogisticRegression(max_iter=250)

    lr_performances = []
    cnn_performances = []

    # Run ML
    for feature_set in feature_sets:

        ############################################# Partitioning #############################################

        X_train, X_test, y_train, y_test = train_test_split(feature_set, y, test_size = .10, random_state=42)

        ############################################# Flattening #############################################
        
        X_train_flat = [img.flatten() for img in X_train]
        X_test_flat = [img.flatten() for img in X_test]

        ############################################# Train + predict #############################################
        
        # Train and predict LR
        lr.fit(X_train_flat, y_train)
        lr_predictions = lr.predict_proba(X_test_flat)

        # Train and predict CNN
        X_train_cnn = np.expand_dims(X_train, -1) # Make sure each img has dimensions 28, 28, 1
        X_test_cnn = np.expand_dims(X_test, -1)
        y_train_cnn = keras.utils.np_utils.to_categorical(y_train, num_classes) # One-hot encoding of y 
        y_test_cnn = keras.utils.np_utils.to_categorical(y_test, num_classes)
        cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # Compile the model
        cnn.fit(X_train_cnn, y_train_cnn, epochs=epochs, verbose = False) # Fit the model
        cnn_predictions = cnn.predict(X_test_cnn) # Make predictions (outcome is in probabilites)
        #cnn_predictions = [np.argmax(cnn_prediction) for cnn_prediction in cnn_predictions] # to go from list of probabilities [0.8, 0.143, 0.03, ...] to the index of the highest probability

        # ############################################# Validate #############################################
        # lr_performance = {
        # "confusion_matrix": metrics.confusion_matrix(y_test, lr_predictions),
        # "classification_report": pd.DataFrame.from_dict(metrics.classification_report(y_test, lr_predictions, output_dict=True))
        # }

        # cnn_performance = {
        # "confusion_matrix": metrics.confusion_matrix(y_test, cnn_predictions),
        # "classification_report": pd.DataFrame.from_dict(metrics.classification_report(y_test, cnn_predictions, output_dict=True))
        # }

        # lr_performances.append(lr_performance)
        # cnn_performances.append(cnn_performance)
        lr_performances.append(lr_predictions)
        cnn_performances.append(cnn_predictions)
    
    # ############################################# Saving performance metrics #############################################
    performances = {}

    for i in range(len(lr_performances)):
        performances[f"{feature_set_names[i]}"] = {"lr": lr_performances[i], "cnn": cnn_performances[i]}
    
    # return performances
    return performances, #lr_performances, cnn_performances
