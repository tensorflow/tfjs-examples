/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * TensorFlow.js Example: LSTM Text Generation
 *
 * Based on Python Keras example:
 *   https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
 */

import * as tf from '@tensorflow/tfjs';
import embed from 'vega-embed';

const testText = document.getElementById('test-text');
const trainModelButton = document.getElementById('train-model');

const sampleText = `PREFACE


SUPPOSING that Truth is a woman--what then? Is there not ground
for suspecting that all philosophers, in so far as they have been
dogmatists, have failed to understand women--that the terrible
seriousness and clumsy importunity with which they have usually paid
their addresses to Truth, have been unskilled and unseemly methods for
winning a woman? Certainly she has never allowed herself to be won; and
at present every kind of dogma stands with sad and discouraged mien--IF,
indeed, it stands at all! For there are scoffers who maintain that it
has fallen, that all dogma lies on the ground--nay more, that it is at
its last gasp. But to speak seriously, there are good grounds for hoping
that all dogmatizing in philosophy, whatever solemn, whatever conclusive
and decided airs it has assumed, may have been only a noble puerilism
and tyronism; and probably the time is at hand when it will be once
and again understood WHAT has actually sufficed for the basis of such
imposing and absolute philosophical edifices as the dogmatists have
hitherto reared: perhaps some popular superstition of immemorial time
(such as the soul-superstition, which, in the form of subject- and
ego-superstition, has not yet ceased doing mischief): perhaps some
play upon words, a deception on the part of grammar, or an
audacious generalization of very restricted, very personal, very
human--all-too-human facts. The philosophy of the dogmatists, it is to
be hoped, was only a promise for thousands of years afterwards, as was
astrology in still earlier times, in the service of which probably more
labour, gold, acuteness, and patience have been spent than on any
actual science hitherto: we owe to it, and to its "super-terrestrial"
pretensions in Asia and Egypt, the grand style of architecture. It seems
that in order to inscribe themselves upon the heart of humanity with
everlasting claims, all great things have first to wander about the
earth as enormous and awe-inspiring caricatures: dogmatic philosophy has
been a caricature of this kind--for instance, the Vedanta doctrine in
Asia, and Platonism in Europe. Let us not be ungrateful to it, although
it must certainly be confessed that the worst, the most tiresome,
and the most dangerous of errors hitherto has been a dogmatist
error--namely, Plato's invention of Pure Spirit and the Good in Itself.
But now when it has been surmounted, when Europe, rid of this nightmare,
can again draw breath freely and at least enjoy a healthier--sleep,
we, WHOSE DUTY IS WAKEFULNESS ITSELF, are the heirs of all the strength
which the struggle against this error has fostered. It amounted to
the very inversion of truth, and the denial of the PERSPECTIVE--the
fundamental condition--of life, to speak of Spirit and the Good as Plato
spoke of them; indeed one might ask, as a physician: "How did such a
malady attack that finest product of antiquity, Plato? Had the wicked
Socrates really corrupted him? Was Socrates after all a corrupter of
youths, and deserved his hemlock?" But the struggle against Plato,
or--to speak plainer, and for the "people"--the struggle against
the ecclesiastical oppression of millenniums of Christianity (FOR
CHRISTIANITY IS PLATONISM FOR THE "PEOPLE"), produced in Europe
a magnificent tension of soul, such as had not existed anywhere
previously; with such a tensely strained bow one can now aim at the
furthest goals. As a matter of fact, the European feels this tension as
a state of distress, and twice attempts have been made in grand style to
unbend the bow: once by means of Jesuitism, and the second time by means
of democratic enlightenment--which, with the aid of liberty of the press
and newspaper-reading, might, in fact, bring it about that the spirit
would not so easily find itself in "distress"! (The Germans invented
gunpowder--all credit to them! but they again made things square--they
invented printing.) But we, who are neither Jesuits, nor democrats,
nor even sufficiently Germans, we GOOD EUROPEANS, and free, VERY free
spirits--we have it still, all the distress of spirit and all the
tension of its bow! And perhaps also the arrow, the duty, and, who
knows? THE GOAL TO AIM AT....

Sils Maria Upper Engadine, JUNE, 1885.




CHAPTER I. PREJUDICES OF PHILOSOPHERS


1. The Will to Truth, which is to tempt us to many a hazardous
enterprise, the famous Truthfulness of which all philosophers have
hitherto spoken with respect, what questions has this Will to Truth not
laid before us! What strange, perplexing, questionable questions! It is
already a long story; yet it seems as if it were hardly commenced. Is
it any wonder if we at last grow distrustful, lose patience, and turn
impatiently away? That this Sphinx teaches us at last to ask questions
ourselves? WHO is it really that puts questions to us here? WHAT really
is this "Will to Truth" in us? In fact we made a long halt at the
question as to the origin of this Will--until at last we came to an
absolute standstill before a yet more fundamental question. We inquired
about the VALUE of this Will. Granted that we want the truth: WHY NOT
RATHER untruth? And uncertainty? Even ignorance? The problem of the
value of truth presented itself before us--or was it we who presented
ourselves before the problem? Which of us is the Oedipus here? Which
the Sphinx? It would seem to be a rendezvous of questions and notes of
interrogation. And could it be believed that it at last seems to us as
if the problem had never been propounded before, as if we were the first
to discern it, get a sight of it, and RISK RAISING it? For there is risk
in raising it, perhaps there is no greater risk.

2. "HOW COULD anything originate out of its opposite? For example, truth
out of error? or the Will to Truth out of the will to deception? or the
generous deed out of selfishness? or the pure sun-bright vision of the
wise man out of covetousness? Such genesis is impossible; whoever dreams
of it is a fool, nay, worse than a fool; things of the highest
value must have a different origin, an origin of THEIR own--in this
transitory, seductive, illusory, paltry world, in this turmoil of
delusion and cupidity, they cannot have their source. But rather in
the lap of Being, in the intransitory, in the concealed God, in the
'Thing-in-itself--THERE must be their source, and nowhere else!"--This
mode of reasoning discloses the typical prejudice by which
metaphysicians of all times can be recognized, this mode of valuation
is at the back of all their logical procedure; through this "belief" of
theirs, they exert themselves for their "knowledge," for something that
is in the end solemnly christened "the Truth." The fundamental belief of
metaphysicians is THE BELIEF IN ANTITHESES OF VALUES. It never occurred
even to the wariest of them to doubt here on the very threshold (where
doubt, however, was most necessary); though they had made a solemn`;

function getTrainingData(textIndices, vocabSize, maxLen, step, startStep, numSteps) {
  const textLen = textIndices.length;
  const xTensorBuffers = [];
  const yTensorBuffers = [];
  let exampleCount = 0;
  for (let i = step * startStep; i < textLen - maxLen - 1; i += step) {
    if (++exampleCount > numSteps) {
      break;
    }

    const xTensorBuffer = new tf.TensorBuffer([1, maxLen, vocabSize]);
    for (let j = 0; j < maxLen; ++j) {
      xTensorBuffer.set(1, 0, j, textIndices[i + j]);
    }
    xTensorBuffers.push(xTensorBuffer);

    const yTensorBuffer = new tf.TensorBuffer([1, vocabSize]);
    yTensorBuffer.set(1, 0, textIndices[i + maxLen]);
    yTensorBuffers.push(yTensorBuffer);
  }

  const xs = tf.concat(xTensorBuffers.map(xBuffer => xBuffer.toTensor()), 0);
  const ys = tf.concat(yTensorBuffers.map(yBuffer => yBuffer.toTensor()), 0);
  return [xs, ys];
}

function prepareData(text, maxLen, step) {
  const charSet = getCharSet(text);
  const textIndices = textToIndices(text, charSet);
  const vocabSize = charSet.length;

  // TODO(cais); DO NOT hardcode 512.
  return [
    getTrainingData(textIndices, vocabSize, maxLen, step, 0, 512), charSet];
}

(async function() {
  testText.value = sampleText;
})();

function createModel(maxLen, charSetSize) {
  const model = tf.sequential();
  model.add(tf.layers.lstm({units: 128, inputShape: [maxLen, charSetSize]}));
  model.add(tf.layers.dense({units: charSetSize, activation: 'softmax'}));

  return model;
}

/**
 * Randomly shuffle an Array.
 * @param {Array} array
 * @returns {Array} Shuffled array.
 */
function shuffle(array) {
  // Origin of the code:
  // https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
  let currentIndex = array.length;
  let temporaryValue;
  let randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {
    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }
  return array;
}

/**
 * Draw one sample from a multinomial distribution.
 * @param {number[]} probs Probabilities. Assumed to sum to 1.
 * @returns {number} A zero-based sample index.
 */
function sampleOneFromMultinomial(probs) {
  const score = Math.random();
  let cumProb = 0;
  const n = probs.length;
  for (let i = 0; i < n; ++i) {
    if (score >= cumProb && score < cumProb + probs[i]) {
      return i;
    }
    cumProb += probs[i];
  }
  return n - 1;
}

// TODO(cais): Use textData with nextEpochData method, randomized.
class TextDataSet {
  constructor(textString, sampleLen, sampleStep) {
    this._textString = textString;
    this._textLen = textString.length;
    this._sampleLen = sampleLen;
    this._sampleStep = sampleStep;

    this._getCharSet();
    this._textToIndices();
    this._generateExampleBeginIndices();
  }

  textLen() {
    return this._textLen;
  }

  charSetSize() {
    return this._charSetSize;
  }

  nextDataEpoch(numExamples) {
    const xsBuffer = new tf.TensorBuffer([
        numExamples, this._sampleLen, this._charSetSize]);
    const ysBuffer  = new tf.TensorBuffer([numExamples, this._charSetSize]);
    for (let i = 0; i < numExamples; ++i) {
      const beginIndex = this._exampleBeginIndices[
          this._examplePosition % this._exampleBeginIndices.length];
      if (i === 0) {
        console.log(beginIndex);
      }
      for (let j = 0; j < this._sampleLen; ++j) {
        xsBuffer.set(1, i, j, this._indices[beginIndex + j]);
      }
      ysBuffer.set(1, i, this._indices[beginIndex + this._sampleLen]);
      this._examplePosition++;
    }
    return [xsBuffer.toTensor(), ysBuffer.toTensor()];
  }

  generateText(model, length, temperature) {
    return tf.tidy(() => {
      const startIndex =
          Math.round(Math.random() * (this._textLen - this._sampleLen - 1));
      let generated = '';
      const sentence =
          this._textString.slice(startIndex, startIndex + this._sampleLen);
      console.log('Generating with seed: ' + sentence);  // DEBUG
      let sentenceIndices = Array.from(
          this._indices.slice(startIndex, startIndex + this._sampleLen));
      console.log(sentenceIndices);  // DEBUG

      // for (let n = 0; n < 8; ++n) {  // DEBUG
      while (generated.length < length) {
        const inputBuffer =
            new tf.TensorBuffer([1, this._sampleLen, this._charSetSize]);
        for (let i = 0; i < this._sampleLen; ++i) {
          inputBuffer.set(1, 0, i, sentenceIndices[i]);
        }
        const input = inputBuffer.toTensor();
        const output = model.predict(input).dataSync();
        input.dispose();
        const winnerIndex = this._sample(output, temperature);  // DEBUG
        const winnerChar = this._charSet[winnerIndex];
        console.log(`${winnerIndex}: ${winnerChar}`);  // DEBUG

        generated += winnerChar;
        sentenceIndices = sentenceIndices.slice(1);
        sentenceIndices.push(winnerIndex);
      }
      return generated;
    });
  }

  _sample(preds, temperature) {
    const logPreds = preds.map(pred => Math.log(pred) / temperature);
    const expPreds = logPreds.map(logPred => Math.exp(logPred));
    let sumExpPreds = 0;
    for (const expPred of expPreds) {
      sumExpPreds += expPred;
    }
    preds = expPreds.map(expPred => expPred / sumExpPreds);
    // Treat preds a the probabilites of a multinomial distribution and
    // randomly draw a sample from the distribution.
    return sampleOneFromMultinomial(preds);
  }

  /**
   * Get the set of unique characters from text.
   */
  _getCharSet() {
    this._charSet = [];
    for (let i = 0; i < this._textLen; ++i) {
      if (this._charSet.indexOf(this._textString[i]) === -1) {
        this._charSet.push(this._textString[i]);
      }
    }
    this._charSetSize = this._charSet.length;
  }

  /**
   * Convert text string to integers.
   */
  _textToIndices() {
    this._indices = new Uint16Array(this._textLen);
    for (let i = 0; i < this._textLen; ++i) {
      this._indices[i] = this._charSet.indexOf(this._textString[i]);
    }
  }

  /**
   * Generate the example-begin indices; shuffle them randomly.
   */
  _generateExampleBeginIndices() {
    // Prepare beginning indices of examples.
    const exampleBeginIndices = [];
    for (let i = 0;
        i < this._textLen - this._sampleLen - 1;
        i += this._sampleStep) {
      exampleBeginIndices.push(i);
    }

    // Randomly shuffle the beginning indices.
    this._exampleBeginIndices = shuffle(exampleBeginIndices);
    this._examplePosition = 0;
  }
};


trainModelButton.addEventListener('click', async () => {
  const sampleLen = 40;
  const sampleStep = 3;

  const textData = testText.value;

  const textDataSet = new TextDataSet(textData, sampleLen, sampleStep);
  console.log(`textLen = ${textDataSet.textLen()}`);  // DEBUG
  console.log(`charSetSize = ${textDataSet.charSetSize()}`);  // DEBUG

  // for (let i = 0; i < 10; ++i) {
  //   const [xs, ys] =  textDataSet.nextDataEpoch(1024);
  //   console.log(xs.shape);  // DEBUG
  //   console.log(ys.shape);  // DEBUG
  //   xs.dispose();
  //   ys.dispose();
  // }

  const model = createModel(sampleLen, textDataSet.charSetSize());
  const learningRate = 0.01;
  const optimzer = tf.train.rmsprop(learningRate);
  model.compile({optimizer: optimzer, loss: 'categoricalCrossentropy'});
  model.summary();

  const trainIterations = 4;
  const epochsPerIteration = 10;
  const epochSize = 2048;
  const batchSize = 128;
  for (let i = 0; i < trainIterations; ++i) {
    const [xs, ys] =  textDataSet.nextDataEpoch(epochSize);
    await model.fit(xs, ys, {
      epochs: epochsPerIteration,
      batchSize: batchSize,
      callbacks: {
        onEpochEnd: async (epochs, log) => {
          console.log(
              `iteration ${i + 1}/${trainIterations}; ` +
              `epoch ${epochs + 1}/${epochsPerIteration}, ` +
              `log = ${JSON.stringify(log)}`);
          await tf.nextFrame();
        },
      }
    });
    xs.dispose();
    ys.dispose();
  }

  const sentence = textDataSet.generateText(model, 100, 0.5);
  console.log(`Generate sentence: ${sentence}`);  // DEBUG
});

// class CharacterTable {
//   /**
//    * Constructor of CharacterTable.
//    * @param chars A string that contains the characters that can appear
//    *   in the input.
//    */
//   constructor(chars) {
//     this.chars = chars;
//     this.charIndices = {};
//     this.indicesChar = {};
//     this.size = this.chars.length;
//     for (let i = 0; i < this.size; ++i) {
//       const char = this.chars[i];
//       if (this.charIndices[char] != null) {
//         throw new Error(`Duplicate character '${char}'`);
//       }
//       this.charIndices[this.chars[i]] = i;
//       this.indicesChar[i] = this.chars[i];
//     }
//   }

//   /**
//    * Convert a string into a one-hot encoded tensor.
//    *
//    * @param str The input string.
//    * @param numRows Number of rows of the output tensor.
//    * @returns The one-hot encoded 2D tensor.
//    * @throws If `str` contains any characters outside the `CharacterTable`'s
//    *   vocabulary.
//    */
//   encode(str, numRows) {
//     const buf = tf.buffer([numRows, this.size]);
//     for (let i = 0; i < str.length; ++i) {
//       const char = str[i];
//       if (this.charIndices[char] == null) {
//         throw new Error(`Unknown character: '${char}'`);
//       }
//       buf.set(1, i, this.charIndices[char]);
//     }
//     return buf.toTensor().as2D(numRows, this.size);
//   }

//   encodeBatch(strings, numRows) {
//     const numExamples = strings.length;
//     const buf = tf.buffer([numExamples, numRows, this.size]);
//     for (let n = 0; n < numExamples; ++n) {
//       const str = strings[n];
//       for (let i = 0; i < str.length; ++i) {
//         const char = str[i];
//         if (this.charIndices[char] == null) {
//           throw new Error(`Unknown character: '${char}'`);
//         }
//         buf.set(1, n, i, this.charIndices[char]);
//       }
//     }
//     return buf.toTensor().as3D(numExamples, numRows, this.size);
//   }

//   /**
//    * Convert a 2D tensor into a string with the CharacterTable's vocabulary.
//    *
//    * @param x Input 2D tensor.
//    * @param calcArgmax Whether to perform `argMax` operation on `x` before
//    *   indexing into the `CharacterTable`'s vocabulary.
//    * @returns The decoded string.
//    */
//   decode(x, calcArgmax = true) {
//     return tf.tidy(() => {
//       if (calcArgmax) {
//         x = x.argMax(1);
//       }
//       const xData = x.dataSync();  // TODO(cais): Performance implication?
//       let output = '';
//       for (const index of Array.from(xData)) {
//         output += this.indicesChar[index];
//       }
//       return output;
//     });
//   }
// }

// /**
//  * Generate examples.
//  *
//  * Each example consists of a question, e.g., '123+456' and and an
//  * answer, e.g., '579'.
//  *
//  * @param digits Maximum number of digits of each operand of the
//  * @param numExamples Number of examples to generate.
//  * @param invert Whether to invert the strings in the question.
//  * @returns The generated examples.
//  */
// function generateData(digits, numExamples, invert) {
//   const digitArray = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
//   const arraySize = digitArray.length;

//   const output = [];
//   const maxLen = digits + 1 + digits;

//   const f = () => {
//     let str = '';
//     while (str.length < digits) {
//       const index = Math.floor(Math.random() * arraySize);
//       str += digitArray[index];
//     }
//     return Number.parseInt(str);
//   };

//   const seen = new Set();
//   while (output.length < numExamples) {
//     const a = f();
//     const b = f();
//     const sorted = b > a ? [a, b] : [b, a];
//     const key = sorted[0] + '`' + sorted[1];
//     if (seen.has(key)) {
//       continue;
//     }
//     seen.add(key);

//     // Pad the data with spaces such that it is always maxLen.
//     const q = `${a}+${b}`;
//     const query = q + ' '.repeat(maxLen - q.length);
//     let ans = (a + b).toString();
//     // Answer can be of maximum size `digits + 1`.
//     ans += ' '.repeat(digits + 1 - ans.length);

//     if (invert) {
//       throw new Error('invert is not implemented yet');
//     }
//     output.push([query, ans]);
//   }
//   return output;
// }

// function convertDataToTensors(data, charTable, digits) {
//   const maxLen = digits + 1 + digits;
//   const questions = data.map(datum => datum[0]);
//   const answers = data.map(datum => datum[1]);
//   return [
//     charTable.encodeBatch(questions, maxLen),
//     charTable.encodeBatch(answers, digits + 1),
//   ];
// }

// function createAndCompileModel(
//     layers, hiddenSize, rnnType, digits, vocabularySize) {
//   const maxLen = digits + 1 + digits;

//   const model = tf.sequential();
//   switch (rnnType) {
//     case 'SimpleRNN':
//       model.add(tf.layers.simpleRNN({
//         units: hiddenSize,
//         recurrentInitializer: 'glorotNormal',
//         inputShape: [maxLen, vocabularySize]
//       }));
//       break;
//     case 'GRU':
//       model.add(tf.layers.gru({
//         units: hiddenSize,
//         recurrentInitializer: 'glorotNormal',
//         inputShape: [maxLen, vocabularySize]
//       }));
//       break;
//     case 'LSTM':
//       model.add(tf.layers.lstm({
//         units: hiddenSize,
//         recurrentInitializer: 'glorotNormal',
//         inputShape: [maxLen, vocabularySize]
//       }));
//       break;
//     default:
//       throw new Error(`Unsupported RNN type: '${rnnType}'`);
//   }
//   model.add(tf.layers.repeatVector({n: digits + 1}));
//   switch (rnnType) {
//     case 'SimpleRNN':
//       model.add(tf.layers.simpleRNN({
//         units: hiddenSize,
//         recurrentInitializer: 'glorotNormal',
//         returnSequences: true
//       }));
//       break;
//     case 'GRU':
//       model.add(tf.layers.gru({
//         units: hiddenSize,
//         recurrentInitializer: 'glorotNormal',
//         returnSequences: true
//       }));
//       break;
//     case 'LSTM':
//       model.add(tf.layers.lstm({
//         units: hiddenSize,
//         recurrentInitializer: 'glorotNormal',
//         returnSequences: true
//       }));
//       break;
//     default:
//       throw new Error(`Unsupported RNN type: '${rnnType}'`);
//   }
//   model.add(tf.layers.timeDistributed(
//       {layer: tf.layers.dense({units: vocabularySize})}));
//   model.add(tf.layers.activation({activation: 'softmax'}));
//   model.compile({
//     loss: 'categoricalCrossentropy',
//     optimizer: 'adam',
//     metrics: ['accuracy']
//   });
//   return model;
// }

// class AdditionRNNDemo {
//   constructor(digits, trainingSize, rnnType, layers, hiddenSize) {
//     // Prepare training data.
//     const chars = '0123456789+ ';
//     this.charTable = new CharacterTable(chars);
//     console.log('Generating training data');
//     const data = generateData(digits, trainingSize, false);
//     const split = Math.floor(trainingSize * 0.9);
//     this.trainData = data.slice(0, split);
//     this.testData = data.slice(split);
//     [this.trainXs, this.trainYs] =
//         convertDataToTensors(this.trainData, this.charTable, digits);
//     [this.testXs, this.testYs] =
//         convertDataToTensors(this.testData, this.charTable, digits);
//     this.model = createAndCompileModel(
//         layers, hiddenSize, rnnType, digits, chars.length);
//   }

//   async train(iterations, batchSize, numTestExamples) {
//     const lossValues = [];
//     const accuracyValues = [];
//     const examplesPerSecValues = [];
//     for (let i = 0; i < iterations; ++i) {
//       const beginMs = performance.now();
//       const history = await this.model.fit(this.trainXs, this.trainYs, {
//         epochs: 1,
//         batchSize,
//         validationData: [this.testXs, this.testYs],
//       });
//       const elapsedMs = performance.now() - beginMs;
//       const examplesPerSec = this.testXs.shape[0] / (elapsedMs / 1000);
//       const trainLoss = history.history['loss'][0];
//       const trainAccuracy = history.history['acc'][0];
//       const valLoss = history.history['val_loss'][0];
//       const valAccuracy = history.history['val_acc'][0];
//       document.getElementById('trainStatus').textContent =
//           `Iteration ${i}: train loss = ${trainLoss.toFixed(6)}; ` +
//           `train accuracy = ${trainAccuracy.toFixed(6)}; ` +
//           `validation loss = ${valLoss.toFixed(6)}; ` +
//           `validation accuracy = ${valAccuracy.toFixed(6)} ` +
//           `(${examplesPerSec.toFixed(1)} examples/s)`;

//       lossValues.push({'epoch': i, 'loss': trainLoss, 'set': 'train'});
//       lossValues.push({'epoch': i, 'loss': valLoss, 'set': 'validation'});
//       embed(
//           '#lossCanvas', {
//             '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
//             'data': {'values': lossValues},
//             'mark': 'line',
//             'encoding': {
//               'x': {'field': 'epoch', 'type': 'ordinal'},
//               'y': {'field': 'loss', 'type': 'quantitative'},
//               'color': {'field': 'set', 'type': 'nominal'},
//             },
//             'width': 400,
//           },
//           {});
//       accuracyValues.push(
//           {'epoch': i, 'accuracy': trainAccuracy, 'set': 'train'});
//       accuracyValues.push(
//           {'epoch': i, 'accuracy': valAccuracy, 'set': 'validation'});
//       embed(
//           '#accuracyCanvas', {
//             '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
//             'data': {'values': accuracyValues},
//             'mark': 'line',
//             'encoding': {
//               'x': {'field': 'epoch', 'type': 'ordinal'},
//               'y': {'field': 'accuracy', 'type': 'quantitative'},
//               'color': {'field': 'set', 'type': 'nominal'},
//             },
//             'width': 400,
//           },
//           {});
//       examplesPerSecValues.push({'epoch': i, 'examples/s': examplesPerSec});
//       embed(
//           '#examplesPerSecCanvas', {
//             '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
//             'data': {'values': examplesPerSecValues},
//             'mark': 'line',
//             'encoding': {
//               'x': {'field': 'epoch', 'type': 'ordinal'},
//               'y': {'field': 'examples/s', 'type': 'quantitative'},
//             },
//             'width': 400,
//           },
//           {});

//       if (this.testXsForDisplay == null ||
//           this.testXsForDisplay.shape[0] !== numTestExamples) {
//         if (this.textXsForDisplay) {
//           this.textXsForDisplay.dispose();
//         }
//         this.testXsForDisplay = this.testXs.slice(
//             [0, 0, 0],
//             [numTestExamples, this.testXs.shape[1], this.testXs.shape[2]]);
//       }

//       const examples = [];
//       const isCorrect = [];
//       tf.tidy(() => {
//         const predictOut = this.model.predict(this.testXsForDisplay);
//         for (let k = 0; k < numTestExamples; ++k) {
//           const scores =
//               predictOut
//                   .slice(
//                       [k, 0, 0], [1, predictOut.shape[1], predictOut.shape[2]])
//                   .as2D(predictOut.shape[1], predictOut.shape[2]);
//           const decoded = this.charTable.decode(scores);
//           examples.push(this.testData[k][0] + ' = ' + decoded);
//           isCorrect.push(this.testData[k][1].trim() === decoded.trim());
//         }
//       });

//       const examplesDiv = document.getElementById('testExamples');
//       while (examplesDiv.firstChild) {
//         examplesDiv.removeChild(examplesDiv.firstChild);
//       }
//       for (let i = 0; i < examples.length; ++i) {
//         const exampleDiv = document.createElement('div');
//         exampleDiv.textContent = examples[i];
//         exampleDiv.className = isCorrect[i] ? 'answer-correct' : 'answer-wrong';
//         examplesDiv.appendChild(exampleDiv);
//       }

//       await tf.nextFrame();
//     }
//   }
// }

// async function runAdditionRNNDemo() {
//   document.getElementById('trainModel').addEventListener('click', async () => {
//     const digits = +(document.getElementById('digits')).value;
//     const trainingSize = +(document.getElementById('trainingSize')).value;
//     const rnnTypeSelect = document.getElementById('rnnType');
//     const rnnType =
//         rnnTypeSelect.options[rnnTypeSelect.selectedIndex].getAttribute(
//             'value');
//     const layers = +(document.getElementById('rnnLayers')).value;
//     const hiddenSize = +(document.getElementById('rnnLayerSize')).value;
//     const batchSize = +(document.getElementById('batchSize')).value;
//     const trainIterations = +(document.getElementById('trainIterations')).value;
//     const numTestExamples = +(document.getElementById('numTestExamples')).value;

//     // Do some checks on the user-specified parameters.
//     const status = document.getElementById('trainStatus');
//     if (digits < 1 || digits > 5) {
//       status.textContent = 'digits must be >= 1 and <= 5';
//       return;
//     }
//     const trainingSizeLimit = Math.pow(Math.pow(10, digits), 2);
//     if (trainingSize > trainingSizeLimit) {
//       status.textContent =
//           `With digits = ${digits}, you cannot have more than ` +
//           `${trainingSizeLimit} examples`;
//       return;
//     }

//     const demo =
//         new AdditionRNNDemo(digits, trainingSize, rnnType, layers, hiddenSize);
//     await demo.train(trainIterations, batchSize, numTestExamples);
//   });
// }

// runAdditionRNNDemo();
