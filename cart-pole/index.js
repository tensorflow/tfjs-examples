import * as tf from '@tensorflow/tfjs';
import {CartPole} from './cart_pole';
import { max } from '@tensorflow/tfjs';

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
    this.oneTensor_ = tf.scalar(1);

    // this.gradFn = tf.variableGrads(this.getCrossEntropyAndSaveActions);
  }

  getGradientsAndSaveActions(inputTensor) {
    const f = () => tf.tidy(() => {
      const [logits, actions] = this.getLogitsAndActions_(inputTensor);
      console.log(`actions:`, actions.shape); actions.print();  // DEBUG
      this.currentActions_ = actions.dataSync();
      const labels = this.oneTensor_.sub(
          tf.tensor2d(this.currentActions_, actions.shape));
      console.log(`labels:`); labels.print();  // DEBUG
      const crossEntropy =
          tf.sigmoidCrossEntropyWithLogits(labels, logits).asScalar();
      console.log(`crossEntropy:`); crossEntropy.print();  // DEBUG
      return crossEntropy;
    });
    return tf.variableGrads(f);
  }

  getCurrentActions() {
    return this.currentActions_;
  }

  // getCrossEntropyAndSaveActions(inputs) {
  //   const [logits, actions] = this.getLogitsAndActions_(inputs);


  //   // TODO(cais): Confirm correctness.

  //   return
  // }

  /**
   * Get action based  on a state tensor.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @returns {Float32Array} 0-1 action values for all the examples in the batch,
   *   length = batchSize.
   */
  getLogitsAndActions_(inputs) {
    return tf.tidy(() => {
      const logits = this.model_.predict(inputs);

      // Get the probability of the left word action.
      const leftProb = tf.sigmoid(logits);
      // Probabilites of the left and right actions.
      const leftRightProbs =
          tf.concat([leftProb, this.oneTensor_.sub(leftProb)], 1);
      leftRightProbs.print();  // DEBUG
      const actions = tf.multinomial(leftRightProbs, 1, null, true);
      return [logits, actions];
    });
  }

  discountAndNormalizeRewards

  train(cartPoleSystem, optimizer, discountRate, numGames, maxStepsPerGame) {
    const allGradients = [];
    const allRewards = [];
    for (let i = 0; i < numGames; ++i) {
      cartPoleSystem.setRandomState();
      const gameRewards = [];
      const gameGradients = [];
      for (let j = 0; j < maxStepsPerGame; ++j) {
        const inputTensor = cartPoleSystem.getStateTensor();
        const gradients =
            this.getGradientsAndSaveActions(inputTensor).grads;
        inputTensor.dispose();

        gameGradients.push(gradients);
        const action = this.currentActions_[0];
        console.log(`j = ${j}, action = ${action}`);
        const isDone = cartPoleSystem.update(action);
        cartPoleSystem.render(cartPoleCanvas);
        if (isDone) {
          console.log('Done!');
          gameRewards.push(0);
          break;
        } else {
          gameRewards.push(1);
        }
      }
      allGradients.push(gameGradients);
      allRewards.push(gameRewards);
      // TODO(cais): Dispose all gradient tensors.
    }
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
  // const inputTensor = cartPole.getStateTensor();
  // const out = policyNet.getGradientsAndSaveActions(inputTensor);
  // console.log(`out:`, out);  // DEBUG
  // // crossEntropy.print();  // DEBUG
  // // TODO(cais): Do not use private member.
  // const [state, done] = cartPole.update(policyNet.getCurrentActions()[0]);
  // console.log(`done = ${done}`);  // DEBUG
  // cartPole.render(cartPoleCanvas);

  const discountRate = 0.95;
  const numGames = 1;
  const maxStepsPerGame = 200;


  policyNet.train(cartPole, null, discountRate, numGames, maxStepsPerGame);

});
