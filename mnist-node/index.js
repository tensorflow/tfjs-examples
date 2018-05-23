const MnistDataset = require('./data');

async function run() {
  const data = new MnistDataset();
  data.loadData();
}

run();