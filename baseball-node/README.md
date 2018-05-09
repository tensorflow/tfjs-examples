# TensorFlow.js Example: Training a baseball model in Node.js 

This demo demonstrates how to train a server-side model to classify baseball pitch types using [Node.js](https://github.com/tensorflow/tfjs-node).

It has four parts:
1. Baseball sensor training and test data.
2. Two ML models that do classification given the sensor data:
   - Model that predicts the type of pitch.
   - Model that predicts if there was a strike.
2. Node.js server that trains a model and serves results over a web socket.
3. Web application that displays pitch type learning statistics.


## Running the Demo
First, prepare the environment and download the baseball training and test data:
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

## Baseball Models

This demo contains two models. The first is a pitch-type model used in the actual client/server architecture. The other model learns how to call balls and strikes like a major-league umpire. It currently does not have any presentation UI but exists for developers to experiment with.

1. Pitch type model - Classifies 7 different pitch types looking at baseball sensor data (pitch-type-model.ts)
2. Strike zone model - A model that can learns how to call balls and strikes based on historical umpire calls (strike-zone-model.ts).
