/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {SnakeGame} from './snake_game';

/**
 * Render the state of the snake game on an HTML canvas element.
 *
 * @param {HTMLCanvasElement} canvas The canvas to render the game in.
 * @param {SnakeGame} game The game to render.
 * @param {Float32Array} qValues Q-values of the current step, optional.
 */
export function renderSnakeGame(canvas, game, qValues) {
  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, width, height);

  const state = game.getState();

  const gameWidth= game.width;
  const gameHeight = game.height;
  const gridWidth = width / gameWidth;
  const gridHeight = height / gameHeight;

  // Draw the grid.
  ctx.strokeStyle = '#aaa';
  ctx.lineWidth = '0';
  for (let i = 0; i <= gameHeight; ++i) {
    ctx.moveTo(0, i * gridHeight);
    ctx.lineTo(width, i * gridHeight);
    ctx.stroke();
  }
  for (let i = 0; i <= gameWidth; ++i) {
    ctx.moveTo(i * gridWidth, 0);
    ctx.lineTo(i * gridWidth, height);
    ctx.stroke();
  }

  // Draw the snake.
  state.s.forEach((yx, i) => {
    const [y, x] = yx;
    ctx.fillStyle = i === 0 ? 'orange' : 'blue';
    ctx.beginPath();
    ctx.rect(x * gridWidth, y * gridHeight, gridWidth, gridHeight);
    ctx.fill();

    if (i === 0) {
      ctx.strokeStyle = 'black';
      ctx.lineWidth = '2';
      ctx.beginPath();
      ctx.rect(
          (x + 0.25) * gridWidth, (y + 0.25) * gridHeight,
          gridWidth * 0.5, gridHeight * 0.5);
      ctx.stroke();
    }
  });

  // Draw the fruit.
  state.f.forEach(yx => {
    const [y, x] = yx;
    ctx.fillStyle = 'green';
    ctx.beginPath();
    ctx.rect(x * gridWidth, y * gridHeight, gridWidth, gridHeight);
    ctx.fill();

    ctx.strokeStyle = 'black';
    ctx.lineWidth = '2';
    ctx.beginPath();
    ctx.arc(
        (x + 0.5) * gridWidth, (y + 0.5) * gridHeight, gridWidth * 0.25,
        0, 2 * Math.PI);
    ctx.stroke();
  });

  if (qValues != null) {   // If qNet is provided, render the q-values.
    if (qValues.length !== 4) {
      throw new Error(`Expected qValues to be of length-4`);
    }
    const [headY, headX] = state.s[0];
    ctx.font = '13px sans serif';
    ctx.fillStyle = 'magenta';
    ctx.beginPath();
    // Left.
    ctx.fillText(
        qValues[0].toFixed(1),
        (headX - 0.9) * gridWidth, (headY + 0.55) * gridHeight);
    // Up.
    ctx.fillText(
        qValues[1].toFixed(1),
        (headX + 0.15) * gridWidth, (headY - 0.45) * gridHeight);
    // Right.
    ctx.fillText(
        qValues[2].toFixed(1),
        (headX + 1.1) * gridWidth, (headY + 0.55) * gridHeight);
    // Down.
    ctx.fillText(
        qValues[3].toFixed(1),
        (headX + 0.15) * gridWidth, (headY + 1.45) * gridHeight);
    ctx.fill();
  }
}
