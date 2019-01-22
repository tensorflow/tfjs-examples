require('@tensorflow/tfjs-node');
const argparse = require('argparse');
const sz_model = require('./strike_zone');

async function run() {
  sz_model.model.summary();
  await sz_model.model.fitDataset(sz_model.trainingData, {
    epochs: 20,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs.loss);
      }
    }
  })
}

run().then(() => console.log('Done'));
