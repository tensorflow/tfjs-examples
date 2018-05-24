const tf = require('@tensorflow/tfjs');
// require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-node-gpu');
tf.setBackend('tensorflow');

const timer = require('node-simple-timer');

const MnistDataset = require('./data');
const model = require('./model');

const NUM_EPOCHS = 10;
const BATCH_SIZE = 100;
const TEST_SIZE = 50;

const data = new MnistDataset();  // TODO - return instance instead.
const totalTimer = new timer.Timer();

async function trainEpoch() {
  let step = 0;
  const stepTimer = new timer.Timer();
  while (data.hasMoreTrainingData()) {
    stepTimer.start();

    const batch = data.nextTrainBatch(BATCH_SIZE);
    const history = await model.fit(
        batch.image, batch.label, {batchSize: BATCH_SIZE, shuffle: false});

    stepTimer.end();
    if (step % 20 === 0) {
      console.log(`  - step: ${step}: loss: ${history.history.loss[0]}, time: ${
          stepTimer.milliseconds()}ms`);
    }
    step++;
  }
}

async function test() {
  if (!data.hasMoreTestData()) {
    data.resetTest();
  }
  const evalData = data.nextTestBatch(TEST_SIZE);
  const output = model.predict(evalData.image);
  const predictions = Array.from(output.argMax(1).dataSync());
  const labels = Array.from(evalData.label.argMax(1).dataSync());

  let correct = 0;
  for (let i = 0; i < TEST_SIZE; i++) {
    if (predictions[i] === labels[i]) {
      correct++;
    }
  }

  const accuracy = ((correct / TEST_SIZE) * 100).toFixed(2);
  console.log(`  * Test set accuracy: ${accuracy}%`);
}

async function run() {
  totalTimer.start();
  await data.loadData();

  const epochTimer = new timer.Timer();
  for (let i = 0; i < NUM_EPOCHS; i++) {
    epochTimer.start();
    await trainEpoch();
    epochTimer.end();
    data.resetTraining();

    console.log();
    console.log(`  * End Epoch: ${i + 1}: time: ${epochTimer.seconds()} secs`);
    test();
    console.log();
  }

  totalTimer.end();
  console.log(
      `  **** Trained ${NUM_EPOCHS} epochs in ${totalTimer.seconds()} secs`);
}

run();
