import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
import { IMAGENET_CLASSES } from './imagenet_classes';

// Where to load the model from.
const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 2;

function clickMenuFn(info, tab) {
  imageClassifier.analyzeImage(info.srcUrl, tab.id);
}

// Add a right-click menu option to trigger classifying the image.
// Menu option should only appear when right-clicking an image.
chrome.contextMenus.create({
  title: "Classify image with TensorFlow.js ", 
  contexts:["image"], 
  onclick: clickMenuFn
});

// Async loads a mobilenet on construction.  Subsequently handles
// requests to classify images through the .analyzeImage API.
// Successful requests will post a chrome message with
// 'IMAGE_CLICK_PROCESSED' action, which the content.js can
// hear and use to manipulate the DOM.
class ImageClassifier {

  constructor() {
    this.loadModel();
  }

  // Loads mobilenet from URL and keeps a reference to it in the object.
  async loadModel() {
    console.log('Loading model...');
    const startTime = performance.now();
    try {
      this.model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
      this.model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

      const totalTime = Math.floor(performance.now() - startTime);
      console.log(`Model loaded and initialized in ${totalTime}ms...`);
    } catch {
      console.error(`Unable to load model from URL: ${MOBILENET_MODEL_PATH}`);
    }
  }

  async analyzeImage(url, tabId) {
    if (!tabId) {
      console.log('No tab.  No prediction.');
      return;
    }
    if (!this.model) {
      console.log('Waiting for model to load...');
      setTimeout(() => { this.analyzeImage(url) }, 5000);
      return;
    }
    const img = await this.loadImage(url);
    if (!img) {
      console.log('could not load image');
      return;
    }
    const predictions = await this.predict(img);
    if (!predictions) {
      console.log('failed to create predictions.');
      return;
    }
    const message = {
      action: 'IMAGE_CLICK_PROCESSED',
      url,
      predictions
    };
    chrome.tabs.sendMessage(tabId, message);
  }

  async loadImage(src) {
    return new Promise(resolve => {
      var img = document.createElement('img');
      img.crossOrigin = "anonymous";
      img.onerror = function(e) {
        resolve(null);
      };
      img.onload = function(e) {
        if ((img.height && img.height > 128) || (img.width && img.width > 128)) {
          // Set image size for tf!
          img.width = IMAGE_SIZE;
          img.height = IMAGE_SIZE;
          resolve(img);
        }
        // Let's skip all tiny images
        resolve(null);
      }
      img.src = src;
    });
  }

  async getTopKClasses(logits, topK) {
    const values = await logits.data();
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: IMAGENET_CLASSES[topkIndices[i]],
        probability: topkValues[i]
      })
    }
    return topClassesAndProbs;
  }


  async predict(imgElement) {
    console.log('Predicting...');
    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    const startTime1 = performance.now();
    // The second start time excludes the extraction and preprocessing and
    // includes only the predict() call.
    let startTime2;
    const logits = tf.tidy(() => {
      const img = tf.browser.fromPixels(imgElement).toFloat();
      const offset = tf.scalar(127.5);
      const normalized = img.sub(offset).div(offset);
      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
      startTime2 = performance.now();
      return this.model.predict(batched);
    });

    // Convert logits to probabilities and class names.
    const classes = await this.getTopKClasses(logits, TOPK_PREDICTIONS);
    const totalTime1 = performance.now() - startTime1;
    const totalTime2 = performance.now() - startTime2;
    console.log(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
    return classes;
  }


}

var imageClassifier = new ImageClassifier();
