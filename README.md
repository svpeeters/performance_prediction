# Performance prediction using generalized loss: A case study for video games
This repository contains the source code and the dataset used in the paper "Performance prediction using generalized loss: A case study for video games".
# Contents
The folder model contains the source code of the models and the experiments. The folders and files have the following contents:

| Folder / File | Description |
|--------------------|-------------|
|     data               |       dataset used in the experiments      |
|       experiment             |    experiment and hpyerparameter tuning source code         |
|          callbacks.py          |    TensorFlow2 callbacks used in the models        |
|          dnn.py          |    TensorFlow2 regression models        |
|          helper.py          |    helper methods and classes        |
|          loss_functions.py          |    implementation of generalized loss and monotonic battch penalty term  |
|          transformer.py          |  mutliset transformers   |

# Preliminary actions 
Before the hyperparameter tuner and the experiments can be executed the scenario1.zip and scenario2.zip in the folder `model/data/case_study` must be decompressed.

# Full dataset
The full cleaned FPS dataset is published via openml:
https://www.openml.org/d/42737

