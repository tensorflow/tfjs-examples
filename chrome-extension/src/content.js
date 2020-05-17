/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// class name for all text nodes added by this script.
const TEXT_DIV_CLASSNAME = 'tfjs_mobilenet_extension_text';
// Thresholds for LOW_CONFIDENCE_THRESHOLD and HIGH_CONFIDENCE_THRESHOLD,
// controlling which messages are printed.
const HIGH_CONFIDENCE_THRESHOLD = 0.5;
const LOW_CONFIDENCE_THRESHOLD = 0.1;

/**
 * Produces a short text string summarizing the prediction
 * Input prediction should be a list of {className: string, prediction: float}
 * objects.
 * @param {[{className: string, predictions: number}]} predictions ordered list
 *     of objects, each with a prediction class and score
 */
function textContentFromPrediction(predictions) {
  if (!predictions || predictions.length < 1) {
    return `No prediction 🙁`;
  }
  // Confident.
  if (predictions[0].probability >= HIGH_CONFIDENCE_THRESHOLD) {
    return `😄 ${predictions[0].className}!`;
  }
  // Not Confident.
  if (predictions[0].probability >= LOW_CONFIDENCE_THRESHOLD &&
      predictions[0].probability < HIGH_CONFIDENCE_THRESHOLD) {
    return `${predictions[0].className}?...\n Maybe ${
        predictions[1].className}?`;
  }
  // Very not confident.
  if (predictions[0].probability < LOW_CONFIDENCE_THRESHOLD) {
    return `😕  ${predictions[0].className}????...\n Maybe ${
        predictions[1].className}????`;
  }
}

/**
 *  Returns a list of all DOM image elements pointing to the provided srcUrl.
 * @param {string} srcUrl which url to search for, including 'http(s)://'
 *     prefix.
 * @returns {HTMLElement[]} all img elements pointing to the provided srcUrl
 */
function getImageElementsWithSrcUrl(srcUrl) {
  const imgElArr = Array.from(document.getElementsByTagName('img'));
  const filtImgElArr = imgElArr.filter(x => x.src === srcUrl);
  return filtImgElArr;
}

/**
 * Finds and removes all of the text predictions added by this extension, and
 * removes them from the DOM. Note: This does not undo the containerization.  A
 * cleaner implementation would move the image node back out of the container
 * div.
 */
function removeTextElements() {
  const textDivs = document.getElementsByClassName(TEXT_DIV_CLASSNAME);
  for (const div of textDivs) {
    div.parentNode.removeChild(div);
  }
}



/**
 *  Moves the provided imgNode into a container div, and adds a text div as a
 * peer.  Styles the container div and text div to place the text
 * on top of the image.
 * @param {HTMLElement} imgNode Which image node to write content on.
 * @param {string} textContent What text to write on the image.
 */
function addTextElementToImageNode(imgNode, textContent) {
  const originalParent = imgNode.parentElement;
  const container = document.createElement('div');
  container.style.position = 'relative';
  container.style.textAlign = 'center';
  container.style.color = 'white';
  const text = document.createElement('div');
  text.className = 'tfjs_mobilenet_extension_text';
  text.style.position = 'absolute';
  text.style.top = '50%';
  text.style.left = '50%';
  text.style.transform = 'translate(-50%, -50%)';
  text.style.fontSize = '34px';
  text.style.fontFamily = 'Google Sans,sans-serif';
  text.style.fontWeight = '700';
  text.style.color = 'white';
  text.style.lineHeight = '1em';
  text.style['-webkit-text-fill-color'] = 'white';
  text.style['-webkit-text-stroke-width'] = '1px';
  text.style['-webkit-text-stroke-color'] = 'black';
  // Add the containerNode as a peer to the image, right next to the image.
  originalParent.insertBefore(container, imgNode);
  // Move the imageNode to inside the containerNode;
  container.appendChild(imgNode);
  // Add the text node right after the image node;
  container.appendChild(text);
  text.textContent = textContent;
}

// Add a listener to hear from the content.js page when the image is through
// processing.  The message should contin an action, a url, and predictions (the
// output of the classifier)
//
// message: {action, url, predictions}
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.action === 'IMAGE_CLICK_PROCESSED' && message.url &&
      message.predictions) {
    // Get the list of images with this srcUrl.
    const imgElements = getImageElementsWithSrcUrl(message.url);
    for (const imgNode of imgElements) {
      const textContent = textContentFromPrediction(message.predictions);
      addTextElementToImageNode(imgNode, textContent);
    }
  }
});

// Set up a listener to remove all annotations if the user clicks
// the left mouse button.  Otherwise, they can easily cloud up the
// window.
window.addEventListener('click', clickHandler, false);
/**
 * Removes text elements from DOM on a left click.
 */
function clickHandler(mouseEvent) {
  if (mouseEvent.button == 0) {
    removeTextElements();
  }
}
