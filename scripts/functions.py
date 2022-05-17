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

def ML(feature_sets, feature_set_names, y):
    """Function for performing machine learning. Outputs performance metrics

    Args:
        feature_sets (List): List of feature sets. The elements in the list contains 4D arrays of dim = (batch, 1st dim of image, 2nd dim of image, 1)
        feature_set_names (_type_): List of the names of the feature sets
        y (np.ndarray): Array containing true value of image

    Returns:
        dict: Nested dictionary with:
            1st level keys = feature_set_names, 
            2nd level keys = lr, svm, cnn
            3rd level keys = confusion_matrix, classification_report
            
            e.g.:
            dict_keys(['X', 'X_bina', 'X_GoL'])
            dict_keys(['lr', 'svm', 'cnn'])
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

    # Define SVM model
    clf = svm.SVC()


    lr_performances = []
    svm_performances = []
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
        lr_predictions = lr.predict(X_test_flat)

        # Train and predict SVM
        clf.fit(X_train_flat, y_train)
        svm_predictions = clf.predict(X_test_flat)

        # Train and predict CNN
        X_train_cnn = np.expand_dims(X_train, -1) # Make sure each img has dimensions 28, 28, 1
        X_test_cnn = np.expand_dims(X_test, -1)
        y_train_cnn = keras.utils.np_utils.to_categorical(y_train, num_classes) # One-hot encoding of y 
        y_test_cnn = keras.utils.np_utils.to_categorical(y_test, num_classes)
        cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # Compile the model
        cnn.fit(X_train_cnn, y_train_cnn, epochs=epochs, verbose = False) # Fit the model
        cnn_predictions = cnn.predict(X_test_cnn) # Make predictions (outcome is in probabilites)
        cnn_predictions = [np.argmax(cnn_prediction) for cnn_prediction in cnn_predictions] # to go from list of probabilities [0.8, 0.143, 0.03, ...] to the index of the highest probability

        ############################################# Validate #############################################
        lr_performance = {
        "confusion_matrix": metrics.confusion_matrix(y_test, lr_predictions),
        "classification_report": pd.DataFrame.from_dict(metrics.classification_report(y_test, lr_predictions, output_dict=True))
        }
        
        svm_performance = {
        "confusion_matrix": metrics.confusion_matrix(y_test, svm_predictions),
        "classification_report": pd.DataFrame.from_dict(metrics.classification_report(y_test, svm_predictions, output_dict=True))
        }

        cnn_performance = {
        "confusion_matrix": metrics.confusion_matrix(y_test, cnn_predictions),
        "classification_report": pd.DataFrame.from_dict(metrics.classification_report(y_test, svm_predictions, output_dict=True))
        }

        lr_performances.append(lr_performance)
        svm_performances.append(svm_performance)
        cnn_performances.append(cnn_performance)
    
    ############################################# Saving performance metrics #############################################
    performances = {}

    for i in range(len(lr_performances)):
        performances[f"{feature_set_names[i]}"] = {"lr": lr_performances[i], "svm": svm_performances[i], "cnn": cnn_performances[i]}
    
    return performances