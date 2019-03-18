# TensorFlow.js Example: Gomoku reinforcement learning

This demo showcases
- training a reinforcement learning model to win a board game.


## Training the model


```sh
yarn
yarn TODO(BILESCHI) BETTER ADD SOMETHING HERE.
```

By default, the training happens on the CPU using the Eigen ops from tfjs-node.
If you have a CUDA-enabled GPU and the necessary drivers and libraries (CUDA and
CuDNN) installed, you can train the model using the CUDA/CuDNN ops from
tfjs-node-gpu. For that, just add the `--gpu` flag:

```sh
yarn
yarn train-rnn --gpu
```

The training code is in the file [train-rnn.js](./train-rnn.js).

## Credits & Thanks
- gomoku implementation is translated from junxiaosong's version at https://github.com/junxiaosong/AlphaZero_Gomoku
