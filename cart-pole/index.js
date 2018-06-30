import * as tf from '@tensorflow/tfjs';
import {CartPole} from './cart_pole';

/**
 * Policy network for controlling the cart-pole system.
 */
class PolicyNetwork {
  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number} hiddenLayerSize Size of the hidden layer.
   */
  constructor(hiddenLayerSize) {
    this.model_ = tf.sequential();
    this.model_.add(tf.layers.dense({
      units: hiddenLayerSize,
      activation: 'elu',
      inputShape: [4]
    }));
    this.model_.add(tf.layers.dense({units: 1}));
  }

  /**
   * Get action based  on a state tensor.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @returns {Float32Array} 0-1 action values for all the examples in the batch,
   *   length = batchSize.
   */
  getActions(inputs) {
    return tf.tidy(() => {
      const output = this.model_.predict(inputs);

      // Get the probability of the left word action.
      const leftProb = tf.sigmoid(output);
      // Probabilites of the left and right actions.
      const leftRightProbs =
          tf.concat([leftProb, tf.sub(tf.onesLike(leftProb), leftProb)], 1);
      leftRightProbs.print();  // DEBUG
      const actions = tf.multinomial(leftRightProbs, 1, null, true).dataSync();
      return actions;
    });
  }
}

const cartPoleCanvas = document.getElementById('cart-pole-canvas');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');
const stepButton = document.getElementById('step');
const cartPole = new CartPole(true);
console.log(cartPole);  // DEBUG

cartPole.render(cartPoleCanvas);

leftButton.addEventListener('click', () => {
  cartPole.update(-1);
  cartPole.render(cartPoleCanvas);
});

rightButton.addEventListener('click', () => {
  cartPole.update(1);
  cartPole.render(cartPoleCanvas);
});


const policyNet =  new PolicyNetwork(5);

stepButton.addEventListener('click', () => {
  const action = policyNet.getActions(cartPole.getStateTensor())[0];
  cartPole.update(action);
  cartPole.render(cartPoleCanvas);
});
