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
    this.oneTensor_ = tf.scalar(1);
  }

  getGradientsAndSaveActions(inputTensor) {
    const f = () => tf.tidy(() => {
      const [logits, actions] = this.getLogitsAndActions_(inputTensor);
      this.currentActions_ = actions.dataSync();
      const labels = this.oneTensor_.sub(
          tf.tensor2d(this.currentActions_, actions.shape));
      return tf.sigmoidCrossEntropyWithLogits(labels, logits).asScalar();
    });
    return tf.variableGrads(f);
  }

  getCurrentActions() {
    return this.currentActions_;
  }

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
      const actions = tf.multinomial(leftRightProbs, 1, null, true);
      return [logits, actions];
    });
  }

  async train(cartPoleSystem,
              optimizer,
              discountRate,
              numGames,
              maxStepsPerGame) {
      const allGradients = [];
      const allRewards = [];
      const gameSteps = [];
      // let totalStepCounter = 0;
      for (let i = 0; i < numGames; ++i) {
        cartPoleSystem.setRandomState();
        const gameRewards = [];
        const gameGradients = [];
        for (let j = 0; j < maxStepsPerGame; ++j) {
          const gradients = tf.tidy(() => {
            const inputTensor = cartPoleSystem.getStateTensor();
            return this.getGradientsAndSaveActions(inputTensor).grads;
          });

          this.pushGradients_(gameGradients, gradients);
          const action = this.currentActions_[0];
          const isDone = cartPoleSystem.update(action);
          if (isDone) {
            console.log('Done!');
            gameRewards.push(0);
            break;
          } else {
            gameRewards.push(1);
          }
          if (j >= maxStepsPerGame) {
            break;
          }
        }
        gameSteps.push(gameRewards.length);
        this.pushGradients_(allGradients, gameGradients);
        allRewards.push(gameRewards);
        await tf.nextFrame();
      }
      console.log(`game steps = ${gameSteps}, mean = ${mean(gameSteps)}`);

      tf.tidy(() => {
        const normalizedRewards =
            discountAndNormalizeRewards(allRewards, discountRate);
        const gradientsToApply =
            scaleAndAverageGradients(allGradients, normalizedRewards);
        optimizer.applyGradients(gradientsToApply);
      });
      tf.dispose(allGradients);
  }

  pushGradients_(record, gradients) {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key]);
      } else {
        record[key] = [gradients[key]];
      }
    }
  }
}

function mean(xs) {
  return xs.reduce((x, prev) => prev + x) / xs.length;
}

function discountRewards(rewards, discountRate) {
  const discounted = [];
  for (let i = rewards.length - 1; i >=0; --i) {
    const reward = rewards[i];
    const prevReward =
        discounted.length > 0 ? discounted[discounted.length - 1] : 0;
    discounted.push(discountRate * prevReward + reward);
  }
  discounted.reverse();
  return discounted;
}

function discountAndNormalizeRewards(rewardSequences, discountRate) {
  return tf.tidy(() => {
    const discounted = [];
    for (const sequence of rewardSequences) {
      discounted.push(discountRewards(sequence, discountRate))
    }

    // Compute the overall mean and stddev.
    const flattened = [];
    for (const sequence of discounted) {
      flattened.push(...sequence);
    }
    const [mean, std] = tf.tidy(() => {
      const r = tf.tensor1d(flattened);
      const mean = tf.mean(r);
      const std = tf.sqrt(tf.mean(tf.square(r.sub(mean))));
      return [mean.dataSync()[0], std.dataSync()[0]];
    });

    // TODO(cais): Maybe normalized should be a tf.Tensor.
    const normalized = [];
    for (const rs of discounted) {
      normalized.push(rs.map(r => (r - mean) / std));
    }
    return normalized;
  });
}

const cartPoleCanvas = document.getElementById('cart-pole-canvas');
const numIterationsInput = document.getElementById('num-iterations');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');
const trainButton = document.getElementById('train');
const cartPole = new CartPole(true);

cartPole.render(cartPoleCanvas);

leftButton.addEventListener('click', () => {
  cartPole.update(-1);
  cartPole.render(cartPoleCanvas);
});

rightButton.addEventListener('click', () => {
  cartPole.update(1);
  cartPole.render(cartPoleCanvas);
});

function scaleAndAverageGradients(allGradients, normalizedRewards) {
  return tf.tidy(() => {
    const rewardScalars = [];
    for (const rewardSequence of normalizedRewards) {
      const rewardScalarSequence = rewardSequence.map(r => tf.scalar(r));
      rewardScalars.push(rewardScalarSequence);
    }

    // TODO(cais): Use tighter tidy() scopes.
    const gradients = {};

    for (const varName in allGradients) {
      const varGradients = allGradients[varName];

      const numGames = varGradients.length;
      gradients[varName] = tf.tidy(() => {
        let numGradients = 0;
        let sum = tf.zerosLike(varGradients[0][0]);
        for (let g = 0; g < numGames; ++g) {
          const numSteps = varGradients[g].length;
          for (let s = 0; s < numSteps; ++s) {
            // TODO(cais): Use broadcasting, vectorized multiplication for
            //   performance?
            const scaledGradients =
                varGradients[g][s].mul(rewardScalars[g][s]);
            sum = sum.add(scaledGradients);
            numGradients++;
          }
        }
        return sum.div(tf.scalar(numGradients));
      });
    }
    return gradients;
  });
}

const policyNet =  new PolicyNetwork(5);

trainButton.addEventListener('click', async () => {
  const trainIterations = Number.parseInt(numIterationsInput.value);
  // TODO(cais): Value sanity checks.
  const discountRate = 0.95;
  const numGames = 20;
  const maxStepsPerGame = 200;
  const learningRate = 0.05;

  const optimizer = tf.train.adam(learningRate);

  for (let i = 0; i < trainIterations; ++i) {
    await policyNet.train(
        cartPole, optimizer, discountRate, numGames, maxStepsPerGame);
    await tf.nextFrame();
  }
});
