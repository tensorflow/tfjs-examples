# TensorFlow.js Example: Reinforcement Learning with Cart-Pole Simulation

## Overview

This example illustrates how to use TensorFlow.js to perform simple
reinforcement learning (RL). Specifically, it showcases an implementation
of the policy-gradient method in TensorFlow.js with a combination of the Layers
and gradients API. This implementation is used to solve the classic cart-pole
control problem, which was originally proposed in:

- Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
  Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
  Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983
- Sutton, "Temporal Aspects of Credit Assignment in Reinforcement Learning",
  Ph.D. Dissertation, Department of Computer and Information Science,
  University of Massachusetts, Amherst, 1984.

It later became one of OpenAI's gym environmnets:
  https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

For an overview of policy gradient methods, see:
  http://www.scholarpedia.org/article/Policy_gradient_methods

### Features:

- Allows user to specify the architecture of the policy network, in particular,
  the number of the neural networks's layers and their sizes (# of units).
- Allows training of the policy network in the browser, optionally with
  simultaneous visualization of the cart-pole system.
- Allows testing in the browser, with visualization.
- Allows saving the policy network to the browser's IndexedDB. The saved policy
  network can later be loaded back for testing and/or further training.

## Usage

```sh
yarn && yarn watch
```
