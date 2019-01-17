const tf = require('@tensorflow/tfjs-node');
const sz_model = require('./strike_zone_model');

async function run() {
  // TODO - add test/eval metrics!
  sz_model.model.summary();
  sz_model.model.fitDataset(sz_model.trainingData, {
    epochs: 20,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs.loss);
      }
    }
  })
}

run().then(() => console.log('Done'));
