# XGBoostvsNeuralNetwork
Program written for comparing XGBoost and NN in learning a given function depending on two parameters.

Given f(x:float ,y: float ) -> float and a square domain in x and y we use a Neural Network and XGBoost to learn this function.

This was written for a presentation given at the Heilbronn doctoral programme summer school in 2022.

We use this to show that it takes a Neural network a significantly longer period of time to learn this function to the same error. 

The code is in the following sections:
1. Data preprocessing
2. Preparing the Neural network
3. Training the Neural network
4. Training XGBoost
5. Reformating the data and models to allow them to be plotted as functions
6. Plot errors and display time needed to learn function from XGBoost and NN

