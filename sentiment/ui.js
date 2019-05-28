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

const exampleReviews = {
  'positive':
      'die hard mario fan and i loved this game br br this game starts slightly boring but trust me it\'s worth it as soon as you start your hooked the levels are fun and exiting they will hook you OOV your mind turns to mush i\'m not kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it\'ll be worth it br br story 9 9 action 10 1 it\'s that good OOV 10 attention OOV 10 average 10',
  'negative':
      'the mother in this movie is reckless with her children to the point of neglect i wish i wasn\'t so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are impossibly involved with the actions on the screen so then why the hell can\'t we have night vision'
};

export function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

export function showMetadata(sentimentMetadataJSON) {
  document.getElementById('modelType').textContent =
      sentimentMetadataJSON['model_type'];
  document.getElementById('vocabularySize').textContent =
      sentimentMetadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      sentimentMetadataJSON['max_len'];
}

export function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('test-example-select');
  testExampleSelect.addEventListener('change', () => {
    setReviewText(exampleReviews[testExampleSelect.value], predict);
  });
  setReviewText(exampleReviews['positive'], predict);
}

export function getReviewText() {
  const reviewText = document.getElementById('review-text');
  return reviewText.value;
}

function doPredict(predict) {
  const reviewText = document.getElementById('review-text');
  const result = predict(reviewText.value);
  status(
      'Inference result (0 - negative; 1 - positive): ' +
      result.score.toFixed(6) +
      ' (elapsed: ' + result.elapsed.toFixed(2) + ' ms)');
}

function setReviewText(text, predict) {
  const reviewText = document.getElementById('review-text');
  reviewText.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const reviewText = document.getElementById('review-text');
  reviewText.addEventListener('input', () => doPredict(predict));
}

export function disableLoadModelButtons() {
  document.getElementById('load-pretrained-remote').style.display = 'none';
  document.getElementById('load-pretrained-local').style.display = 'none';
}
