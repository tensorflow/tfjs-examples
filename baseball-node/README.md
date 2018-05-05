# TensorFlow.js Example: Training Baseball data in Node.js 

### This demo demonstrates using the Node.js bindings for sever-side model training and predictions.

This package contains 3 components:
1. Models and training data for baseball
2. Node.js server for running pitch type model and reporting over socket.io
3. Client for listening to the server and displaying pitch type predictions


## Running the Demo
First, prepare the environment:
```sh
yarn && yarn download-training-data
```

Next, start the client:
```sh
yarn start-client
```

In a new shell, start the server:
```sh
yarn start-server
```

To perform model only training to see how Node.js works with the two models, run the following:
```sh
yarn train-pitch-type-model
```
```sh
yarn train-strike-zone-model
```

## Pitch Models

1. Pitch type model - TODO(kreeger)
2. Strike zone model - TODO(kreeger)