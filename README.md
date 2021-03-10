# Heartbeat-Anomaly-Detection
In this project, we have used a python autoencoder based on the LSTM or Long Short Term Memory algorithm.
LSTM is a Deep Recurrent Neural Network algorithm.
The aim of the project is to get the dataset of the ECG data and split it into anomaly heartbeat types and normal ones.
The prediction is possible because of the use of LSTM's autoencoder, allowing a simpler, more treatable data.
The program is divided into 3 main files: main program, the prediction file which generates the model.pth 
The prediction file contains the classes, the imported packages and the functions that pre-process the dataset and trains the model that is saved in the model.pth
The main files load the model, and applies the predict function, then generating the graphs based on the error measurements, identifying a threshhold that splits the normal hearbeats from the anomalies.
