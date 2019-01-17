const tf = require('@tensorflow/tfjs-node');
const pitch_type = require('./pitch_type');

async function run() {
  // TODO - add test/eval metrics!
  pitch_type.model.summary();
  await pitch_type.model.fitDataset(pitch_type.trainingData, {
    epochs: 2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs.loss);
      }
    }
  });

  // pitch_model.evaluate(false);
}

run();
