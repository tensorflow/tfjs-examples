
import {CartPole} from './cart_pole';

const cartPoleCanvas = document.getElementById('cart-pole-canvas');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');
const cartPole = new CartPole();
console.log(cartPole);  // DEBUG

cartPole.setRandomState();
cartPole.render(cartPoleCanvas);

leftButton.addEventListener('click', () => {
  cartPole.update(-1);
  cartPole.render(cartPoleCanvas);
});

rightButton.addEventListener('click', () => {
  cartPole.update(1);
  cartPole.render(cartPoleCanvas);
});
