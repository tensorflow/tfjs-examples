import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as d3 from 'd3';

const decoderUrl =
  'http://127.0.0.1:8080/decoder/model.json';

let decoder;

const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_CHANNELS = 1;

const LATENT_DIMS = 2;

async function loadModel(modelUrl) {
  const decoder = await tf.loadLayersModel(modelUrl);
  tfvis.show.modelSummary({ name: 'decoder' }, decoder);
  tfvis.show.layer({ name: 'dense2' }, decoder.getLayer('dense_Dense2'));
  tfvis.show.layer({ name: 'dense3' }, decoder.getLayer('dense_Dense3'));
  return decoder;
}

function generateLatentSpace(dimensions, pointsPerDim, start, end) {
  const result = [];
  for (let i = 0; i < dimensions; i++) {
    tf.tidy(() => {
      const values = tf.linspace(start, end, pointsPerDim).dataSync();
      result.push(values);
    });
  }

  return result;
}

function decodeZ(inputTensor) {
  return tf.tidy(() => {
    const batched = decoder.apply(inputTensor).mul(255).cast('int32');
    console.log('batched', batched)
    // const flat = tf.unstack(batched)[0];
    // console.log('flat', flat)
    const reshaped = batched.reshape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]);
    // console.log('reshaped', reshaped)
    return reshaped;
  });
}

async function renderTensorToCanvas(tensor, canvas) {
  return tf.browser.toPixels(tensor, canvas);
}

async function renderLatentSpace(latentSpace) {
  document.getElementById('plot-area').innerText = '';
  // handles only 2d for now
  const rows =
    d3.select('.plot-area').selectAll('div.row').data(latentSpace[0]);

  const rEnter = rows.enter().append('div').attr('class', 'row');
  rows.exit().remove();

  const cols = rEnter.selectAll('div.col').data(latentSpace[1]);
  const cEnter = cols.enter()
    .append('div')
    .attr('class', 'col')
    .append('canvas')
    .attr('width', 50)
    .attr('height', 50);


  console.log('latent space', latentSpace)
  rows.merge(rEnter).each(async function (rowZ, rowIndex) {
    const cols = d3.select(this).selectAll('.col');
    cols.each(async function (colZ, colIndex) {
      const canvas = d3.select(this).select('canvas').node();
      const zTensor = tf.tensor2d([[rowZ, colZ]]);
      const imageTensor = decodeZ(zTensor);

      renderTensorToCanvas(imageTensor, canvas).then(() => {
        tf.dispose([zTensor, imageTensor]);
      });
    });
  });
}

function getParams() {
  const ppd = document.getElementById('pointsPerDim');
  const start = document.getElementById('start');
  const end = document.getElementById('end');

  // console.log(ppdInput.value, start.value, end.value);
  return {
    pointsPerDim: parseInt(ppd.value), start: parseFloat(start.value),
    end: parseFloat(end.value),
  }
}

function draw() {
  const params = getParams();
  console.log('params', params);
  const latentSpace =
    generateLatentSpace(LATENT_DIMS, params.pointsPerDim, params.start, params.end);
  console.log('latentspace', latentSpace);
  renderLatentSpace(latentSpace);
}

async function run() {
  decoder = await loadModel(decoderUrl);
  console.log('decoder', decoder);
  draw();
}

function setupListeners() {
  document.getElementById('update').addEventListener('click', () => {
    draw();
  })
}

(function () {
  setupListeners();
  run();
})();
