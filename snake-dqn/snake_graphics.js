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
 *   If provided, will overlay Q-values for possible actions on the rendered
 *   graph.
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
      ctx.moveTo((x + 0.25) * gridWidth,  (y + 0.5) * gridHeight);
      ctx.lineTo((x + 0.75) * gridWidth,  (y + 0.5) * gridHeight);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(
            (x + 0.5) * gridWidth, (y + 0.5) * gridHeight, gridWidth * 0.25,
            0, Math.PI);
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
    if (qValues.length !== 3) {
      throw new Error(
          `Expected qValues to be of length 3, ` +
          `but got length ${qValues.length}`);
    }
    const [headY, headX] = state.s[0];


    let qW;
    let qN;
    let qE;
    let qS;
    if (game.snakeDirection === 'l') {
      qW = qValues[0];
      qS = qValues[1];
      qN = qValues[2];
    } else if (game.snakeDirection === 'u') {
      qN = qValues[0];
      qW = qValues[1];
      qE = qValues[2];
    } else if (game.snakeDirection === 'r') {
      qE = qValues[0];
      qN = qValues[1];
      qS = qValues[2];
    } else if (game.snakeDirection === 'd') {
      qS = qValues[0];
      qE = qValues[1];
      qW = qValues[2];
    }

    const {qWNormalized, qNNormalized, qENormalized, qSNormalized} =
        normalizeQValuesForDisplay(qW, qN, qE, qS);
    drawQValueOverlay(ctx, qW, qWNormalized,
        (headX - 1) * gridWidth, headY * gridHeight, gridWidth, gridHeight);
    drawQValueOverlay(ctx, qN, qNNormalized,
        headX * gridWidth, (headY - 1) * gridHeight, gridWidth, gridHeight);
    drawQValueOverlay(ctx, qE, qENormalized,
        (headX  + 1) * gridWidth, headY * gridHeight, gridWidth, gridHeight);
    drawQValueOverlay(ctx, qS, qSNormalized,
        headX * gridWidth, (headY + 1) * gridHeight, gridWidth, gridHeight);
  }
}

function normalizeQValuesForDisplay(qW, qN, qE, qS) {
  const scores = [qW, qN, qE, qS].filter(x => x != null);
  const min = Math.min(...scores);
  const max = Math.max(...scores);

  const normalize = q => (q - min) / (max - min);
  return {
    qWNormalized: qW == null ? qW : normalize(qW),
    qNNormalized: qN == null ? qN : normalize(qN),
    qENormalized: qE == null ? qE : normalize(qE),
    qSNormalized: qS == null ? qE : normalize(qS)
  };
}

function drawQValueOverlay(context, q, qNormalized, x, y, w, h) {
  if (q == null) {
    return;
  }
  context.globalAlpha = 0.2;
  let r = Math.floor((1 - qNormalized) * 255);
  let g = 255;
  let b = Math.floor((1 - qNormalized) * 255);
  context.fillStyle = `rgb(${r},${g},${b})`;
  context.beginPath();
  context.rect(x, y, w, h);
  context.fill();
  context.globalAlpha = 1;

  context.font = '13px sans serif';
  r = Math.floor((1 - qNormalized) * 100 + 64);
  g = Math.floor((1 - qNormalized) * 100 + 64);
  b = Math.floor((1 - qNormalized) * 100 + 64);
  context.fillStyle = `rgb(${r},${g},${b})`;
  context.beginPath();
  context.fillText(q.toFixed(1), x + 0.15 * w, y + 0.55 * h);
  context.fill();
}
