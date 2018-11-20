/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require("@tensorflow/tfjs");
// require("@tensorflow/tfjs-node");

// init variable a,b,c,d
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

// build module
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a
      .mul(x.pow(tf.scalar(3))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d); // + d
  });
}

// make MSE check loss
function loss(predictions, labels) {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions
    .sub(labels)
    .square()
    .mean();
  return meanSquareError;
}

// optimizer
function optimizer(predictions, labels) {
  const learningRate = 0.5;
  return tf.train.sgd(learningRate);
}

// train
function train(xs, ys, numIterations = 75) {
  const optimize = optimizer();

  for (let iter = 0; iter < numIterations; iter++) {
    optimize.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}

// generateData
function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a),
      tf.scalar(coeff.b),
      tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    // Generate polynomial data
    const three = tf.scalar(3, "int32");
    const ys = a
      .mul(xs.pow(three))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
      // Add random noise to the generated data
      // to make the problem a bit more interesting
      .add(tf.randomNormal([numPoints], 0, sigma));

    // Normalize the y values to the range 0 to 1.
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys: ysNormalized
    };
  });
}

const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
const trainingData = generateData(100, trueCoefficients);
console.log("---True a,b,c,d:-0.800,-0.200,0.900,0.500");
console.log("---train Before a,b,c,d:");
a.print();
b.print();
c.print();
d.print();
train(trainingData.xs, trainingData.ys, 1000);
console.log("---train After a,b,c,d:");
a.print();
b.print();
c.print();
d.print();
