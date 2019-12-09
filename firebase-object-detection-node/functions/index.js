const functions = require('firebase-functions');
const express = require('express');
const Busboy = require('busboy');
const path = require('path');

const tf = require('@tensorflow/tfjs-node');

const labels = require('./model/new_object_detection_1/assets/labels.json');

const app = express();

let objectDetectionModel;

async function loadModel() {
  // Warm up the model
  if (!objectDetectionModel) {
    objectDetectionModel = await tf.node.loadSavedModel(
        './model/new_object_detection_1', ['serve'], 'serving_default');
  }
  const tempTensor = tf.zeros([1, 2, 2, 3]).toInt();
  objectDetectionModel.predict(tempTensor);
}

app.get('/', async (req, res) => {
  res.sendFile(path.join(__dirname + '/index.html'));
  loadModel();
})

app.post('/predict', async (req, res) => {
  const busboy = new Busboy({headers: req.headers});
  let fileBuffer = new Buffer('');
  req.files = {file: []};

  busboy.on('field', (fieldname, value) => {
    req.body[fieldname] = value;
  });

  busboy.on('file', (fieldname, file, filename, encoding, mimetype) => {
    file.on('data', (data) => {fileBuffer = Buffer.concat([fileBuffer, data])});

    file.on('end', () => {
      const file_object = {
        fieldname,
        'originalname': filename,
        encoding,
        mimetype,
        buffer: fileBuffer
      };

      req.files.file.push(file_object)
    });
  });

  busboy.on('finish', async () => {
    const buf = req.files.file[0].buffer;
    const uint8array = new Uint8Array(buf);

    if (!objectDetectionModel) {
      objectDetectionModel = await tf.node.loadSavedModel(
          './model/new_object_detection_1', ['serve'], 'serving_default');
    }
    const imageTensor = await tf.node.decodeImage(uint8array);
    const input = imageTensor.expandDims(0);

    let outputTensor = objectDetectionModel.predict({'x': input});

    const scores = await outputTensor['detection_scores'].arraySync();
    const boxes = await outputTensor['detection_boxes'].arraySync();
    const names = await outputTensor['detection_classes'].arraySync();
    outputTensor['detection_scores'].dispose();
    outputTensor['detection_boxes'].dispose();
    outputTensor['detection_classes'].dispose();
    outputTensor['num_detections'].dispose();
    const detectedBoxes = [];
    const detectedNames = [];
    for (let i = 0; i < scores[0].length; i++) {
      if (scores[0][i] > 0.3) {
        detectedBoxes.push(boxes[0][i]);
        detectedNames.push(labels[names[0][i]]);
      }
    }
    res.send({boxes: detectedBoxes, names: detectedNames});
  });


  busboy.end(req.rawBody);
  req.pipe(busboy);
});

loadModel();

exports.app = functions.https.onRequest(app);
