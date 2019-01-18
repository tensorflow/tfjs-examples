const argparse = require('argparse');
const tf = require('@tensorflow/tfjs-node');
const pitch_type = require('./pitch_type');

async function run(epochCount, savePath) {
  pitch_type.model.summary();
  await pitch_type.model.fitDataset(pitch_type.trainingData, {
    epochs: epochCount,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs.loss);
      }
    }
  });

  // Eval against test data:
  console.log(pitch_type.model.evaluate(true /* useTestData */));

  if (savePath !== null) {
    await pitch_type.model.save(`file://${savePath}`);
    console.log(`Saved model to path: ${savePath}`);
  }
}

run();


const parser = new argparse.ArgumentParser(
    {description: 'TensorFlow.js Pitch Type Training Example', addHelp: true});
parser.addArgument('--epochs', {
  type: 'int',
  defaultValue: 20,
  help: 'Number of epochs to train the model for.'
})
parser.addArgument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});

const args = parser.parseArgs();
console.log('args', typeof (args.epochs));

run(args.epochs, args.model_save_path)
