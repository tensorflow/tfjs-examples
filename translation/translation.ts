
/**
 * Train a simple LSTM model for character-level language translation.
 * This is based on the Tensorflow.js example at:
 *   https://github.com/tensorflow/tfjs-examples/blob/master/translation/python/translation.py
 *
 * The training data can be downloaded with a command like the following example:
 *   wget http://www.manythings.org/anki/fra-eng.zip
 *
 * Author: Huan LI <zixia@zixia.net>
 * 2018, https://github.com/huan
 *
 */


import fs from 'fs'
import path from 'path'

import {ArgumentParser} from 'argparse'
import readline from 'readline'
import mkdirp from 'mkdirp'

const {zip} = require('zip-array')
const invertKv = require('invert-kv')

import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'

let FLAGS = {} as any

async function readData (
  dataFile: string,
) {
  // Vectorize the data.
  const input_texts: string[] = []
  const target_texts: string[] = []

  const input_characters = new Set<string>()
  const target_characters = new Set<string>()

  const fileStream = fs.createReadStream(dataFile)
  const rl = readline.createInterface({
    input:    fileStream,
    output:   process.stdout,
    terminal: false,
  })

  let lineNumber = 0
  rl.on('line', line => {
    if (++lineNumber > FLAGS.num_samples) {
      rl.close()
      return
    }

    let [input_text, target_text] = line.split('\t')
    // We use "tab" as the "start sequence" character for the targets, and "\n"
    // as "end sequence" character.
    target_text = '\t' + target_text + '\n'

    input_texts.push(input_text)
    target_texts.push(target_text)

    for (const char of input_text) {
      if (!input_characters.has(char)) {
        input_characters.add(char)
      }
    }
    for (const char of target_text) {
      if (!target_characters.has(char)) {
        target_characters.add(char)
      }
    }
  })

  await new Promise(r => rl.on('close', r))

  const input_character_list = [...input_characters].sort()
  const target_character_list = [...target_characters].sort()

  const num_encoder_tokens = input_character_list.length
  const num_decoder_tokens = target_character_list.length

  // Math.max() does not work with very large arrays because of the stack limitation
  const max_encoder_seq_length = input_texts.map(text => text.length)
                                            .reduceRight((prev, curr) => curr > prev ? curr : prev, 0)
  const max_decoder_seq_length = target_texts.map(text => text.length)
                                              .reduceRight((prev, curr) => curr > prev ? curr : prev, 0)

  console.log('Number of samples:', input_texts.length)
  console.log('Number of unique input tokens:', num_encoder_tokens)
  console.log('Number of unique output tokens:', num_decoder_tokens)
  console.log('Max sequence length for inputs:', max_encoder_seq_length)
  console.log('Max sequence length for outputs:', max_decoder_seq_length)

  const input_token_index = input_character_list.reduce(
    (prev, curr, idx) => (prev[curr] = idx, prev),
    {} as {[char: string]: number},
  )
  const target_token_index = target_character_list.reduce(
    (prev, curr, idx) => (prev[curr] = idx, prev),
    {} as {[char: string]: number},
  )

  // Save the token indices to file.
  const metadata_json_path = path.join(
    FLAGS.artifacts_dir,
    'metadata.json',
  )

  if (!fs.existsSync(path.dirname(metadata_json_path))) {
    mkdirp.sync(path.dirname(metadata_json_path))
  }

  const metadata = {
    'input_token_index': input_token_index,
    'target_token_index': target_token_index,
    'max_encoder_seq_length': max_encoder_seq_length,
    'max_decoder_seq_length': max_decoder_seq_length,
  }

  fs.writeFileSync(metadata_json_path, JSON.stringify(metadata))
  console.log('Saved metadata at: ', metadata_json_path)

  const encoder_input_data_buf = tf.buffer<tf.Rank.R3>([
    input_texts.length,
    max_encoder_seq_length,
    num_encoder_tokens,
  ])
  const decoder_input_data_buf = tf.buffer<tf.Rank.R3>([
    input_texts.length,
    max_decoder_seq_length,
    num_decoder_tokens,
  ])
  const decoder_target_data_buf = tf.buffer<tf.Rank.R3>([
    input_texts.length,
    max_decoder_seq_length,
    num_decoder_tokens,
  ])

  for (
    const [i, [input_text, target_text]]
    of (zip(input_texts, target_texts).entries() as IterableIterator<[number, [string, string]]>)
  ) {
    for (const [t, char] of input_text.split('').entries()) {
      // encoder_input_data[i, t, input_token_index[char]] = 1.
      encoder_input_data_buf.set(1, i, t, input_token_index[char])
    }

    for (const [t, char] of target_text.split('').entries()) {
      // decoder_target_data is ahead of decoder_input_data by one timestep
      decoder_input_data_buf.set(1, i, t, target_token_index[char])
      if (t > 0) {
        // decoder_target_data will be ahead by one timestep
        // and will not include the start character.
        decoder_target_data_buf.set(1, i, t - 1, target_token_index[char])
      }
    }
  }

  const encoder_input_data = encoder_input_data_buf.toTensor()
  const decoder_input_data = decoder_input_data_buf.toTensor()
  const decoder_target_data = decoder_target_data_buf.toTensor()

  return {
    input_texts,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    input_token_index,
    target_token_index,
    encoder_input_data,
    decoder_input_data,
    decoder_target_data,
  }

}

function seq2seqModel (
  num_encoder_tokens: number,
  num_decoder_tokens: number,
  latent_dim: number,
) {
  /**
  Create a Keras model for the seq2seq translation.

  Args:
    num_encoder_tokens: Total number of distinct tokens in the inputs
      to the encoder.
    num_decoder_tokens: Total number of distinct tokens in the outputs
      to/from the decoder
    latent_dim: Number of latent dimensions in the LSTMs.

  Returns:
    encoder_inputs: Instance of `keras.Input`, symbolic tensor as input to
      the encoder LSTM.
    encoder_states: Instance of `keras.Input`, symbolic tensor for output
      states (h and c) from the encoder LSTM.
    decoder_inputs: Instance of `keras.Input`, symbolic tensor as input to
      the decoder LSTM.
    decoder_lstm: `keras.Layer` instance, the decoder LSTM.
    decoder_dense: `keras.Layer` instance, the Dense layer in the decoder.
    model: `keras.Model` instance, the entire translation model that can be
      used in training.
  */
  // Define an input sequence and process it.
  const encoder_inputs = tf.layers.input({
    shape: [null, num_encoder_tokens] as number[],
    name: 'encoderInputs',
  })

  const encoder = tf.layers.lstm({
    units: latent_dim,
    returnState: true,
    name: 'encoderLstm',
  })
  const [, state_h, state_c] = encoder.apply(encoder_inputs) as tf.SymbolicTensor[]
  // We discard `encoder_outputs` and only keep the states.
  const encoder_states = [state_h, state_c]

  // Set up the decoder, using `encoder_states` as initial state.
  const decoder_inputs = tf.layers.input({
    shape: [null, num_decoder_tokens] as number[],
    name: 'decoderInputs',
  })
  // We set up our decoder to return full output sequences,
  // and to return internal states as well. We don't use the
  // return states in the training model, but we will use them in inference.
  const decoder_lstm = tf.layers.lstm({
    units: FLAGS.latent_dim,
    returnSequences: true,
    returnState: true,
    name: 'decoderLstm',
  })

  const [decoder_outputs, ] = decoder_lstm.apply(
    [decoder_inputs, ...encoder_states],
  ) as tf.Tensor[]

  const decoder_dense = tf.layers.dense({
    units: num_decoder_tokens,
    activation: 'softmax',
    name: 'decoderDense',
  })

  const decoder_dense_outputs = decoder_dense.apply(decoder_outputs) as tf.SymbolicTensor

  // Define the model that will turn
  // `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  const model = tf.model({
    inputs: [encoder_inputs, decoder_inputs],
    outputs: decoder_dense_outputs,
    name: 'seq2seqModel',
  })
  return {
    encoder_inputs,
    encoder_states,
    decoder_inputs,
    decoder_lstm,
    decoder_dense,
    model,
  }
}

async function decode_sequence (
  input_seq: tf.Tensor,
  encoder_model: tf.Model,
  decoder_model: tf.Model,
  num_decoder_tokens: number,
  target_begin_index: number,
  reverse_target_char_index: {[indice: number]: string},
  max_decoder_seq_length: number,
) {
  /**
  Decode (i.e., translate) an encoded sentence.

  Args:
    input_seq: A `numpy.ndarray` of shape
      `(1, max_encoder_seq_length, num_encoder_tokens)`.
    encoder_model: A `keras.Model` instance for the encoder.
    decoder_model: A `keras.Model` instance for the decoder.
    num_decoder_tokens: Number of unique tokens for the decoder.
    target_begin_index: An `int`: the index for the beginning token of the
      decoder.
    reverse_target_char_index: A lookup table for the target characters, i.e.,
      a map from `int` index to target character.
    max_decoder_seq_length: Maximum allowed sequence length output by the
      decoder.

  Returns:
    The result of the decoding (i.e., translation) as a string.
  """
  */

  // Encode the input as state vectors.
  let states_value = encoder_model.predict(input_seq) as tf.Tensor[]

  // Generate empty target sequence of length 1.
  let target_seq = tf.buffer<tf.Rank.R3>([
    1,
    1,
    num_decoder_tokens,
  ])

  // Populate the first character of target sequence with the start character.
  target_seq.set(1, 0, 0, target_begin_index)

  // Sampling loop for a batch of sequences
  // (to simplify, here we assume a batch of size 1).
  let stop_condition = false
  let decoded_sentence = ''
  while (!stop_condition) {
    const [output_tokens, h, c] = decoder_model.predict(
      [target_seq.toTensor(), ...states_value]
    ) as [
      tf.Tensor<tf.Rank.R3>,
      tf.Tensor<tf.Rank.R2>,
      tf.Tensor<tf.Rank.R2>,
    ]

    // Sample a token
    const sampled_token_index = await output_tokens
                                      .squeeze()
                                      .argMax(-1)
                                      .array() as number

    const sampled_char = reverse_target_char_index[sampled_token_index]
    decoded_sentence += sampled_char

    // Exit condition: either hit max length
    // or find stop character.
    if ( sampled_char === '\n'
      || decoded_sentence.length > max_decoder_seq_length
    ) {
      stop_condition = true
    }

    // Update the target sequence (of length 1).
    target_seq = tf.buffer<tf.Rank.R3>([1, 1, num_decoder_tokens], 'float32')
    target_seq.set(1, 0, 0, sampled_token_index)

    // Update states
    states_value = [h, c]
  }
  return decoded_sentence
}

async function main () {
  const {
    input_texts,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    target_token_index,
    encoder_input_data,
    decoder_input_data,
    decoder_target_data,
  } = await readData(FLAGS.data_path)

  const {
    encoder_inputs,
    encoder_states,
    decoder_inputs,
    decoder_lstm,
    decoder_dense,
    model,
  } = seq2seqModel(
    num_encoder_tokens,
    num_decoder_tokens,
    FLAGS.latent_dim,
  )

  // Run training.
  model.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
  })

  await model.fit(
    [
      encoder_input_data,
      decoder_input_data,
    ],
    decoder_target_data,
    {
      batchSize: FLAGS.batch_size,
      epochs: FLAGS.epochs,
      validationSplit: 0.2,
    },
  )

  // Huan: be aware that the Node need a `file://` prefix to local filename
  await model.save('file://' + FLAGS.artifacts_dir)

  // tfjs.converters.save_keras_model(model, FLAGS.artifacts_dir)

  // Next: inference mode (sampling).
  // Here's the drill:
  // 1) encode input and retrieve initial decoder state
  // 2) run one step of decoder with this initial state
  // and a "start of sequence" token as target.
  // Output will be the next target token
  // 3) Repeat with the current target token and current states

  // Define sampling models
  const encoder_model = tf.model({
    inputs: encoder_inputs,
    outputs: encoder_states,
    name: 'encoderModel',
  })

  const decoder_state_input_h = tf.layers.input({
    shape: [FLAGS.latent_dim],
    name: 'decoderStateInputHidden',
  })
  const decoder_state_input_c = tf.layers.input({
    shape: FLAGS.latent_dim,
    name: 'decoderStateInputCell',
  })
  const decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
  let [decoder_outputs, state_h, state_c] = decoder_lstm.apply(
      [decoder_inputs, ...decoder_states_inputs]
  ) as tf.SymbolicTensor[]

  const decoder_states = [state_h, state_c]
  decoder_outputs = decoder_dense.apply(decoder_outputs) as tf.SymbolicTensor
  const decoder_model = tf.model({
    inputs: [decoder_inputs, ...decoder_states_inputs],
    outputs: [decoder_outputs, ...decoder_states],
    name: 'decoderModel',
  })

  // Reverse-lookup token index to decode sequences back to
  // something readable.
  const reverse_target_char_index = invertKv(target_token_index) as {[indice: number]: string}

  const target_begin_index = target_token_index['\t']

  for (let seq_index = 0; seq_index < FLAGS.num_test_sentences; seq_index++) {
    // Take one sequence (part of the training set)
    // for trying out decoding.
    const input_seq = encoder_input_data.slice(seq_index, 1)

    // Get expected output
    const target_seq_voc = decoder_target_data.slice(seq_index, 1).squeeze([0]) as tf.Tensor2D
    const target_seq_tensor = target_seq_voc.argMax(-1) as tf.Tensor1D

    const target_seq_list = await target_seq_tensor.array()

    // One-hot to index
    const target_seq = target_seq_list.map(indice => reverse_target_char_index[indice])

    // Array to string
    const target_seq_str = target_seq.join('').replace('\n', '')
    const decoded_sentence = await decode_sequence(
        input_seq, encoder_model, decoder_model, num_decoder_tokens,
        target_begin_index, reverse_target_char_index, max_decoder_seq_length)
    console.log('-')
    console.log('Input sentence:', input_texts[seq_index])
    console.log('Target sentence:', target_seq_str)
    console.log('Decoded sentence:', decoded_sentence)
  }
}


const parser = new ArgumentParser({
  version: '0.0.1',
  addHelp: true,
  description: 'Keras seq2seq translation model training and serialization',
})

parser.addArgument(
  ['data_path'],
  {
    type: 'string',
    help: 'Path to the training data, e.g., ~/ml-data/fra-eng/fra.txt',
  },
)
parser.addArgument(
  '--batch_size',
  {
    type: 'int',
    defaultValue: 64,
    help: 'Training batch size.'
  }
)
parser.addArgument(
  '--epochs',
  {
    type: 'int',
    defaultValue: 200,
    help: 'Number of training epochs.',
  },
)
parser.addArgument(
  '--latent_dim',
  {
    type: 'int',
    defaultValue: 256,
    help: 'Latent dimensionality of the encoding space.',
  },
)
parser.addArgument(
  '--num_samples',
  {
    type: 'int',
    defaultValue: 10000,
    help: 'Number of samples to train on.',
  }
)
parser.addArgument(
  '--num_test_sentences',
  {
    type: 'int',
    defaultValue: 100,
    help: 'Number of example sentences to test at the end of the training.',
  },
)
parser.addArgument(
  '--artifacts_dir',
  {
    type: 'string',
    defaultValue: '/tmp/translation.keras',
    help: 'Local path for saving the TensorFlow.js artifacts.',
  },
)

;[FLAGS,] = parser.parseKnownArgs()
main()
