const tf = require('@tensorflow/tfjs');
// require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-node-gpu');
tf.setBackend('tensorflow');

const timer = require('node-simple-timer');

const MnistDataset = require('./data');
const model = require('./model');

const BATCH_SIZE = 100;

async function run() {
  const data = new MnistDataset();
  await data.loadData();


  const epochTimer = new timer.Timer();
  for (let i = 0; i < 10; i++) {
    epochTimer.start();

    let step = 0;
    const stepTimer = new timer.Timer();
    while (data.hasMoreData()) {
      stepTimer.start();

      const batch = data.nextTrainBatch(BATCH_SIZE);
      const history = await model.fit(
          batch.image, batch.label,
          {batchSize: BATCH_SIZE, epochs: 1, shuffle: false});

      stepTimer.end();
      if (step % 20 === 0) {
        console.log(`  step: ${step}: loss: ${history.history.loss[0]}, time: ${
            stepTimer.milliseconds()}ms`);
      }
      step++;
    }

    epochTimer.end();
    data.reset();
    console.log(
        `\nend epoch: ${i}: epoch time: ${epochTimer.seconds()} secs\n`);
  }
}

run();