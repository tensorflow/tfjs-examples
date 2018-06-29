/**
 * Based on: http://incompleteideas.net/book/code/pole.c
 */

export class CartPole {
  constructor() {
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
  }

  setRandomState() {
    this.x = Math.random() - 0.5;
    this.theta = (Math.random() - 0.5) * (Math.PI / 2 * 0.3);
  }

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
