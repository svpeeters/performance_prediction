# Performance prediction for hardware-softwareconfigurations: A case study for video games
This repository contains the source code and the dataset used in the following paper: Peeters, Sven, Vitalik Melnikov, and Eyke Hüllermeier. Performance Prediction for Hardware–Software Configurations: A Case Study for Video Games. Advances in Intelligent Data Analysis XIX: 19th International Symposium on Intelligent Data Analysis, IDA 2021, Porto, Portugal, April 26–28, 2021, Proceedings. Springer Nature.

# Contents
The folder model contains the source code of the models and the experiments. The folders and files have the following contents:

| Folder / File | Description |
|--------------------|-------------|
|     data               |       dataset used in the experiments      |
|       experiment             |    experiment and hpyerparameter tuning source code         |
|          callbacks.py          |    TensorFlow2 callbacks used in the models        |
|          dnn.py          |    TensorFlow2 regression models        |
|          helper.py          |    helper methods and classes        |
|          loss_functions.py          |    implementation of generalized loss and monotonic penalty term  |
|          transformer.py          |  mutliset transformers   |

# Preliminary actions 
Before the hyperparameter tuner and the experiments can be executed the scenario1.zip and scenario2.zip in the folder `model/data/case_study` must be decompressed.

# Full dataset
The full cleaned FPS dataset is published via openml:
https://www.openml.org/d/42737

