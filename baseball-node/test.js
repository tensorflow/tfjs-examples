const pitchData = require('./pitch_data');
const strikeZoneModel = require('./strike_zone_model');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');


function convertSZData(filename) {
  // Write training data:
  const file = fs.createWriteStream(filename + '.csv');
  const data = pitchData.loadPitchData(filename + '.json');
  data.forEach((pitch) => {
    const line = `${pitch.px},${pitch.pz},${pitch.sz_top},${pitch.sz_bot},${
        pitch.left_handed_batter}\n`;
    file.write(line);
  });
  file.close();
}

function convertPTData(filename) {
  const file = fs.createWriteStream(filename + '.csv');
  const data = pitchData.loadPitchData(filename + '.json');
  data.forEach((pitch) => {
    const line = `${pitch.px},${pitch.pz},${pitch.sz_top},${pitch.sz_bot},${
        pitch.left_handed_batter}\n`;
    file.write(line);
  });
  file.close();
}

convertSZData('strike_zone_training_data');
convertSZData('strike_zone_test_data');
convertPTData('pitch_type_training_data');
convertPTData('pitch_type_test_data');
