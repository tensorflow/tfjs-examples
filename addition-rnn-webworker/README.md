# TensorFlow.js Example: Addition RNN in Webworker

This example uses an RNN to compute (in a worker thread) the addition of two integers by doing
string => string translation. Obviously it's not the best way to add two
numbers, but it makes a fun example. In this way, we can do long-running computation without blocking the UI thread.

Note: This example is based on the addition-rnn [example](https://github.com/tensorflow/tfjs-examples/tree/master/addition-rnn) in this repo, which is based on the original Keras python code [here](https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py)

[See this example live!](https://storage.googleapis.com/tfjs-examples/addition-rnn/dist/index.html)
