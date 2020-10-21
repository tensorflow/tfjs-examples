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

import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';

// Messages.
const DOWNLOAD_MESSAGE =
    'This interactive model visualizer will run in the browser: the required TensorFlow.js model files are currently being loaded. Thanks for your patience!';
const NO_MODEL_METADATA_ERROR_MESSAGE =
    'No model metadata URL provided. The visualizer could not start';
const TEST_IMAGE_FETCH_FAILURE_MESSAGE =
    'The test image couldn\'t be fetched, please check the JS console for more details.';

// Constants.
const MAX_NB_RESULTS = 100;
const UNKNOWN_LABEL_DISPLAY_NAME = 'unknown';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})

export class AppComponent implements OnInit {
  title = 'interactive-visualizers';

  // Model related variables.
  publisherThumbnailUrl: string|null = null;
  publisherName: string|null = null;
  modelMetadataUrl: string|null = null;
  modelMetadata: any|null = null;
  modelType: string|null = null;
  model: tf.GraphModel|null = null;
  labelmap: string[]|null = null;
  defaultScoreThreshold = 0.0;

  // Test data related variables.
  testImagesIndexUrl: string|null = null;
  testImages: Array<{imageUrl: string, thumbnailUrl: string}> = [];
  uploadedImages: string[] = [];
  queryImageDataURL: string|null = null;
  imageSelectedIndex: number|null = null;
  isDraggedOver = false;

  // Results variables.
  resultsKeyName: string|null = null;
  resultsValueName: string|null = null;
  classifierResults: Array<{displayName: string, score: number}>|null = null;

  ngOnInit(): void {
    // Sanity checks on URL query parameters.
    const urlParams = new URLSearchParams(window.location.search);
    if (!urlParams.has('modelMetadataUrl')) {
      throw new Error(NO_MODEL_METADATA_ERROR_MESSAGE);
    }
    const modelMetadataUrl = urlParams.get('modelMetadataUrl');
    if (urlParams.has('publisherThumbnailUrl')) {
      this.publisherThumbnailUrl = urlParams.get('publisherThumbnailUrl');
    }
    if (urlParams.has('publisherName')) {
      this.publisherName = urlParams.get('publisherName');
    }

    this.initApp(modelMetadataUrl);
  }

  async fetchModel(modelUrl: string): Promise<tf.GraphModel> {
    const model = await tf.loadGraphModel(modelUrl);
    return model;
  }

  /**
   * Initializes the app for the provided model metadata URL.
   */
  async initApp(modelMetadataUrl: string): Promise<void> {
    // Load model & metadata.
    this.modelMetadataUrl = modelMetadataUrl;
    const metadataResponse = await fetch(this.modelMetadataUrl);
    this.modelMetadata = await metadataResponse.json();
    const modelUrl = this.getAssetsUrlPrefix() + 'model.json';
    this.model = await this.fetchModel(modelUrl);
    if (this.modelMetadata.tfjs_classifier_model_metadata) {
      this.modelType = 'classifier';
    }

    // Fetch test data if any.
    this.testImagesIndexUrl = this.modelMetadata.test_images_index_path;
    try {
      const testImagesResponse = await fetch(this.testImagesIndexUrl);
      const testImageNames = await testImagesResponse.json();
      this.testImages = [];
      for (const testImageName of testImageNames) {
        const testImageExt =
            testImageName.split('.')[testImageName.split('.').length - 1];
        const testImageNamePrefix = testImageName.split('.' + testImageExt)[0];
        this.testImages.push({
          imageUrl: this.getTestImagesUrlPrefix() + testImageName,
          thumbnailUrl: this.getTestImagesUrlPrefix() + testImageNamePrefix +
              '_thumb.' + testImageExt,
        });
      }
      if (this.testImages.length > 0) {
        // If test images are present, automatically run on the first one.
        this.testImageSelected(this.testImages[0].imageUrl, 0);
      }

    } catch (error) {
      console.error(`Couldn't fetch test images: ${error}`);
    }
  }

  /**
   * Get the assets URL path, e.g. `http://assets-url-path/` if the metadata URL
   * is `http://assets-url-path/metadata.json`.
   */
  getAssetsUrlPrefix(): string {
    const metadataFileName = this.modelMetadataUrl.split('/').pop();
    return this.modelMetadataUrl.split(metadataFileName)[0];
  }

  /**
   * Get the test images URL path, e.g. `http://test-images-url-path/` if the
   * test images index is `http://test-images-url-path/index.json`.
   */
  getTestImagesUrlPrefix(): string {
    const imageIndexFileName = this.testImagesIndexUrl.split('/').pop();
    return this.testImagesIndexUrl.split(imageIndexFileName)[0];
  }

  /**
   * On click on a test image.
   */
  async testImageSelected(imageUrl: string, index: number): Promise<void> {
    try {
      this.imageSelectedIndex = index;
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const reader = new FileReader();
      reader.onload = () => {
        let imageDataURL: string = reader.result as string;
        // If MIME is unknown, replace it to 'image/jpg' as an attempt.
        imageDataURL =
            imageDataURL.replace('application/octet-stream', 'image/jpg');
        this.handleInputImage(imageDataURL, index);
      };
      reader.readAsDataURL(blob);
    } catch (error) {
      console.error(
          `Fetching the test image failed with the following error: ${error}`);
      alert(TEST_IMAGE_FETCH_FAILURE_MESSAGE);
    }
  }

  /**
   * Triggered when user uploads image files. Run inference on each image.
   */
  imageFilesSelected(event: InputEvent): void {
    const element = event.target as HTMLInputElement;
    if (element.files.length > 0) {
      for (const imageFile of Array.from(element.files)) {
        if (!imageFile.type.match('image*')) {
          continue;
        }
        this.readImageFile(imageFile);
      }
    }
  }

  /**
   * On drag over event.
   */
  dragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    if (!this.isDraggedOver) {
      this.isDraggedOver = true;
    }
  }

  /**
   * On drag leave event.
   */
  dragLeave(): void {
    this.isDraggedOver = false;
  }

  /**
   * On drop event.
   */
  dragDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDraggedOver = false;

    if (event.dataTransfer.files.length > 0) {
      for (const imageFile of Array.from(event.dataTransfer.files)) {
        this.readImageFile(imageFile);
      }
    }
  }

  readImageFile(imageFile: File): void {
    try {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageDataURL = reader.result as string;
        const imageIndex = this.addUploadedImage(imageDataURL);
        this.imageSelectedIndex = imageIndex;
        this.handleInputImage(imageDataURL, imageIndex);
      };
      reader.readAsDataURL(imageFile);
    } catch (error) {
      console.error(
          `An uploaded image failed to be read with the following error: ${
              error}`);
    }
  }

  /**
   * Adds an uploaded image and returns its index.
   */
  addUploadedImage(imageDataURL: string): number {
    this.uploadedImages.push(imageDataURL);
    return this.testImages.length + this.uploadedImages.length - 1;
  }

  /**
   * Handles an input image as data URL. Displays it in the query element, and
   * sends it for inference to the right handler depending on the model type.
   */
  async handleInputImage(imageDataURL: string, index: number): Promise<void> {
    this.queryImageDataURL = imageDataURL;
    const image = new Image();
    image.onload = async () => {
      switch (this.modelType) {
        case 'classifier':
          const classifierResults = await this.runImageClassifier(image);
          if (this.imageSelectedIndex === index) {
            // Display results only for the last selected image (as the user may
            // have switched selection while inference was running).
            this.classifierResults = classifierResults;
            this.resultsKeyName = 'Type';
            this.resultsValueName = 'Score';
          }
          break;
        default:
          console.error(
              `The model type \`${this.modelType}\ isn't currently supported.`);
      }
    };
    image.src = imageDataURL;
  }

  /**
   * Prepare an input image by converting it to tf.Tensor and resizing it
   * according to the expected model input.
   */
  prepareImageInput(image: HTMLImageElement, inputTensorMetadata: {
    shape: number[]
  }): tf.Tensor {
    return tf.tidy(() => {
      let imageTensor = tf.browser.fromPixels(image, /* numChannels= */ 3);

      // Resize the query image according to the model input shape.
      imageTensor = tf.image.resizeBilinear(
          imageTensor,
          [inputTensorMetadata.shape[1], inputTensorMetadata.shape[2]], false);

      // Map to the correct input shape, range and type. The models expect float
      // inputs in the range [0, 1].
      imageTensor = imageTensor.toFloat().div(255).expandDims(0);

      return imageTensor;
    });
  }

  /**
   * Fetches and parses the labelmap. If IDs are provided, they are used.
   * Otherwise list indices are used as IDs.
   */
  async fetchLabelmap(labelmapName: string): Promise<void> {
    try {
      const labelmapUrl = this.getAssetsUrlPrefix() + labelmapName;
      const labelmapResponse = await fetch(labelmapUrl);
      const labelmapJson = await labelmapResponse.json();
      let maxId = labelmapJson.item.length - 1;
      for (const item of labelmapJson.item) {
        if (item.id && item.id > maxId) {
          maxId = item.id;
        }
      }
      const labelmap = [];
      for (let i = 0; i <= maxId; ++i) {
        labelmap.push(UNKNOWN_LABEL_DISPLAY_NAME);
      }
      for (let i = 0; i < labelmapJson.item.length; ++i) {
        const item = labelmapJson.item[i];
        let displayName = UNKNOWN_LABEL_DISPLAY_NAME;
        if (item.name) {
          displayName = item.name;
        }
        if (item.display_name) {
          displayName = item.display_name;
        }
        if (item.id) {
          labelmap[item.id] = displayName;
        } else {
          labelmap[i] = displayName;
        }
      }
      this.labelmap = labelmap;
    } catch (error) {
      this.labelmap = [];
      console.error(
          `Fetching the labelmap failed with the following error: ${error}`);
    }
  }

  /**
   * Run the model in case of image classification, and return classifier
   * results.
   */
  async runImageClassifier(image: HTMLImageElement):
      Promise<Array<{displayName: string, score: number}>> {
    // Prepare inputs.
    const inputTensorMetadata =
        this.modelMetadata.tfjs_classifier_model_metadata.input_tensor_metadata;
    const imageTensor = this.prepareImageInput(image, inputTensorMetadata);

    // Execute the model.
    const outputTensor: tf.Tensor =
        await this.model.executeAsync(imageTensor) as tf.Tensor;
    tf.dispose(imageTensor);
    const squeezedOutputTensor = outputTensor.squeeze();
    tf.dispose(outputTensor);
    const predictions: number[] =
        await squeezedOutputTensor.array() as number[];
    tf.dispose(squeezedOutputTensor);

    // Fetch the labelmap and score thresholds, then assign labels to the
    // prediction results.
    const outputHeadMetadata = this.modelMetadata.tfjs_classifier_model_metadata
                                   .output_head_metadata[0];
    let scoreThreshold = 0.0;
    if (outputHeadMetadata.score_threshold != null) {
      scoreThreshold = outputHeadMetadata.score_threshold;
    }
    if (this.labelmap == null && outputHeadMetadata.labelmap_path != null) {
      await this.fetchLabelmap(outputHeadMetadata.labelmap_path);
    }
    let results = [];
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] > scoreThreshold) {
        if (this.labelmap != null && this.labelmap.length > i) {
          results.push({
            displayName: this.labelmap[i],
            score: predictions[i],
          });
        } else {
          results.push({
            displayName: UNKNOWN_LABEL_DISPLAY_NAME,
            score: predictions[i],
          });
        }
      }
    }

    // Sort remaining results in descending order.
    results.sort((a, b) => {
      if (a.score > b.score) {
        return -1;
      }
      return 1;
    });

    // Keep a maximum of MAX_NB_RESULTS for the UI.
    if (results.length > MAX_NB_RESULTS) {
      results = results.slice(0, MAX_NB_RESULTS);
    }
    return results;
  }
}
