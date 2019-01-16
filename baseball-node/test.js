const tf = require('@tensorflow/tfjs-node');

const strikeZoneModel = require('./strike_zone_model');
const timers = require('node-simple-timer');

const timer = new timers.Timer();


console.log('model', strikeZoneModel.model);
async function test() {
  timer.start();
  // TODO - fill this out
  timer.end();
  console.log('  > epoch train time: ' + timer.seconds());
}

test();
