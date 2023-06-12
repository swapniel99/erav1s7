# ERA V1 Assignment 7

## Model 1:
### Target:
    - Getting a working modular setup. We will use assignment 5 code.
    - Model.py has models and utils.py has train, test and plotting functions.
### Results:
    - Model has 5.9M parameters.
    - Train Accuracy = 99.95
    - Test Accuracy = 99.35
### Analysis:
    - Our setup is running correctly.
    - Model is very large and overfitting
    - Closer look at images tell us that a receptive field of 5 pixels is enough to catch edges.

## Model 2:
### Target:
    - Getting the model skeleton right.
### Results:
    - Model has 174k parameters.
    - Train Accuracy = 99.7
    - Test Accuracy = 99.34
### Analysis:
    - Simple model gradually increasing channels to 256.
    - Model is still large and overfitting.

## Model 3:
### Target:
    - Making the model as light as possible.
### Results:
    - Model has 7781 parameters.
    - Train Accuracy = 99.11
    - Test Accuracy = 98.96
### Analysis:
    - Simple model gradually increasing channels to 32.
    - Overfitting reduced significantly.

## Model 4:
### Target:
    - Add Batch Normalisation to increase efficiency.
### Results:
    - Model has 7979 parameters.
    - Train Accuracy = 99.34
    - Test Accuracy = 99.24
### Analysis:
    - Model has started overfitting again slightly.

## Model 5:
### Target:
    - Add Dropout of 1%.
### Results:
    - Model has 7979 parameters.
    - Train Accuracy = 99.23
    - Test Accuracy = 99.26
### Analysis:
    - No longer overfitting. Infact it is slighly underfitting now.

## Model 6:
### Target:
    - Add image augmentations random rotation and random perspective.
### Results:
    - Model has 7979 parameters.
    - Train Accuracy = 98.69
    - Test Accuracy = 99.53
### Analysis:
    - Crossed 99.4 for first time and test loss is fluctuating.
    - Time to add an LR scheduler.

## Model 7:
### Target:
    - Add LR Scheduler ReduceLROnPlateau.
### Results:
    - Model has 7979 parameters.
    - Train Accuracy = 98.85
    - Test Accuracy = 99.44
### Analysis:
    - Target achieved: crossed 99.4% validation accuracy 3 times (epochs 11, 14 and 15)
