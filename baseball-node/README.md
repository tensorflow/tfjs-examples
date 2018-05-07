# TensorFlow.js Example: Training a baseball model in Node.js 

This demo demonstrates how to use the [Node.js bindings](https://github.com/tensorflow/tfjs-node) for TensorFlow.js. 

It has four parts:
1. Baseball sensor data
2. Two ML models that do classification given the sensor data:
   - Model that predicts the type of pitch.
   - Model that predicts if there was a strike.
2. Node.js server that trains a model and serves results over a web socket.
3. Web application that displays predictions and training stats.


## Running the Demo
First, prepare the environment and download the baseball data from MLB:
```sh
yarn && yarn download-data
```

Next, start the client:
```sh
yarn start-client
```

Open the client running at: http://localhost:8080/

In a new shell, start the server:
```sh
yarn start-server
```

If you are interested in testing out the training, without running a web server:
```sh
yarn train-pitch-model
```
```sh
yarn train-strike-model
```

## Pitch Models

1. Pitch type model - TODO(kreeger)
2. Strike zone model - TODO(kreeger)