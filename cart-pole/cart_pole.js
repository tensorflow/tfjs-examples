/**
 * Based on: http://incompleteideas.net/book/code/pole.c
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Cart-pole system simulator.
 *
 * The system has four state variables:
 *
 *   - x: The 1D location of the cart.
 *   - xDot: The velocity of the cart.
 *   - theta: The angle of the pole (in radians). A value of 0 corresponds to
 *     a vertical position.
 *   - thetaDot: The angular velocity of the pole.
 *
 * The system is controlled through a single action.
 */
export class CartPole {

  /**
   *
   * @param {bool} initializeStateRandomly Whether to initialize the state
   *   randomlly. If `false` (default), all states will be initialized to
   *   zero.
   */
  constructor(initializeStateRandomly) {
    this.gravity = 9.8;
    this.massCart = 1.0;
    this.massPole = 0.1;
    this.totalMass = this.massCart + this.massPole;
    this.cartWidth = 0.2;
    this.cartHeight = 0.1;
    this.length = 0.5;
    this.poleMoment = this.massPole * this.length;
    this.forceMag = 10.0;
    this.tau = 0.02;  // Seconds between state updates.

    this.x = 0;  // Cart position, meters.
    this.xDot = 0;  // Cart velocity.
    this.theta = 0;  // Pole angle, radians.
    this.thetaDot = 0;  // Pole angle velocity.
    if (initializeStateRandomly) {
      this.setRandomState_();
    }
  }

  /**
   * Set the state of the cart-pole system randomly.
   */
  setRandomState_() {
    this.x = Math.random() - 0.5;
    this.xDot = (Math.random() - 0.5) * 1;
    this.theta = (Math.random() - 0.5) * (Math.PI / 2 * 0.3);
    this.thetaDot =  (Math.random() - 0.5) * 0.5;
  }

  /**
   * Get current state as a tf.Tensor of shape [1, 4].
   */
  getStateTensor() {
    const buffer = new tf.TensorBuffer([1, 4]);
    buffer.set(this.x, 0, 0);
    buffer.set(this.xDot, 0, 1);
    buffer.set(this.theta, 0, 2);
    buffer.set(this.thetaDot, 0, 3);
    return buffer.toTensor();
  }

  /**
   * Update the cart-pole system using an action.
   * @param {number} action Only the sign of `action` matters.
   *   A value > 0 leads to a rightward force of a fixed magnitude.
   *   A value <= 0 leads to a leftward force of the same fixed magnitude.
   */
  update(action) {
    const force = action > 0 ? this.forceMag : -this.forceMag;

    const cosTheta = Math.cos(this.theta);
    const sinTheta = Math.sin(this.theta);

    const temp =
        (force + this.poleMoment * this.thetaDot * this.thetaDot * sinTheta) /
        this.totalMass;
    const thetaAcc =
        (this.gravity * sinTheta - cosTheta * temp) /
        (this.length *
            (4 / 3 - this.massPole * cosTheta * cosTheta / this.totalMass));
    const xAcc = temp - this.poleMoment * thetaAcc * cosTheta / this.totalMass;

    // Update the four state variables, using Euler's metohd.
    this.x += this.tau * this.xDot;
    this.xDot += this.tau * xAcc;
    this.theta += this.tau * this.thetaDot;
    this.thetaDot += this.tau * thetaAcc;

    console.log(
        `x = ${this.x}; xDot = ${this.xDot};\n'` +
        `theta = ${this.theta}; thetaDot = ${this.theta}`);  // DEBUG
  }

  /**
   * Render the current state of the system on an HTML canvas.
   *
   * @param {HTMLCanvasElement} canvas
   */
  render(canvas) {
    const X_MIN = -2;
    const X_MAX = 2;
    const xRange = X_MAX - X_MIN;
    const scale = canvas.width / xRange;

    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    const halfW = canvas.width / 2;

    // 1. Draw the cart.
    const railY = canvas.height * 0.8;
    const cartW = this.cartWidth * scale;
    const cartH = this.cartHeight * scale;

    const cartX = this.x * scale + halfW;

    context.beginPath();
    context.rect(cartX - cartW / 2, railY - cartH / 2, cartW, cartH);
    context.stroke();

    // 2. Draw the pole.
    const angle = this.theta + Math.PI / 2;
    const poleTopX =
        halfW + scale * (this.x + Math.cos(angle) * this.length);
    const poleTopY =
        railY - scale * (this.cartHeight / 2 + Math.sin(angle) * this.length);
    context.beginPath();
    context.moveTo(cartX, railY - cartH / 2);
    context.lineTo(poleTopX, poleTopY);
    context.stroke();
  }
}
