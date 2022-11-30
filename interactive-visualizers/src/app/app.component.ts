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
import { environment } from './../environments/environment';

// Name declaration for TFWeb imported libraries.
declare let ImageClassifierViz: any;
declare let ImageSegmenterViz: any;
declare let ObjectDetectorViz: any;

// Messages.
const DOWNLOAD_MESSAGE =
    'This interactive model visualizer will run in the browser: the required TensorFlow.js model files are currently being loaded. Thanks for your patience!';
const NO_MODEL_ERROR_MESSAGE =
    'No model URL provided. The visualizer could not start';
const NO_TFWEB_API_ERROR_MESSAGE =
    'No TFWeb API selected. The visualizer could not start';
const TEST_IMAGE_FETCH_FAILURE_MESSAGE =
    'The test image couldn\'t be fetched, please check the JS console for more details.';

// Constants.
const DEFAULT_MODEL_DISPLAY_NAME = 'Interactive model visualizer';
const MAX_NB_RESULTS = 100;
const UNKNOWN_LABEL_DISPLAY_NAME = 'unknown';
const DEFAULT_DETECTION_THRESHOLD = 0.2;
const DETECTION_RECTANGLE_BORDER_WIDTH = 2;
const EPSILON = 0.0000001;
const WARMUP_IMAGE_DATA_URL = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAaCAYAAABCfffNAAABdklEQVRIS2P8/vnDfwYaA8ZRS0gJYZzB9enTJ4YDBw8xPHz0iIGRkRGvmf///2eQl5VjcHCwY+Dj48NQi9WSv3//MrR39jD8/PmdQUhIiIGVlRWvJb9//2F49+4dAzs7O0NleSkDMzMzinqslhw5epRh7fqNDP09XaSECkNhSRlDcGAAg421FWFLTp0+wzB56nSGxQvmkmRJbEIyQ252JoOZqcmoJYRDbjS4CIcRkorR4BqY4KqoqgFb3NHWguEAiuMEVBpXVtcyKCoqgg2/f/8+Q3trM4O8nBzcMoosARX5FdW1DKnJSQzJifFgQ+fOX8gwe+48ho7WZgYHezuwGNmWzJu/kGHW3LkM7S3NDI4O9ihBtP/AQYbKmlqGtORkhqTEePIskZeTZbiHJViQbYIFo5KiIsPDR4+JL4U3b9nG0NjcwuDh7sbQ0tRAVAqrqWtg2LFzF0N9bQ2Dr48X4aIepGLtug0MwUEBRFkAU4RLz/BpEgEApSZOND6VlesAAAAASUVORK5CYII=';
/**
 * Uses the Pascal VOC[1] color list (256 colors).
 * [1]: http://host.robots.ox.ac.uk/pascal/VOC/
 */
const COLOR_LIST = [
  [0.000000, 0.000000, 0.000000], [0.501961, 0.000000, 0.000000],
  [0.000000, 0.501961, 0.000000], [0.501961, 0.501961, 0.000000],
  [0.000000, 0.000000, 0.501961], [0.501961, 0.000000, 0.501961],
  [0.000000, 0.501961, 0.501961], [0.501961, 0.501961, 0.501961],
  [0.250980, 0.000000, 0.000000], [0.752941, 0.000000, 0.000000],
  [0.250980, 0.501961, 0.000000], [0.752941, 0.501961, 0.000000],
  [0.250980, 0.000000, 0.501961], [0.752941, 0.000000, 0.501961],
  [0.250980, 0.501961, 0.501961], [0.752941, 0.501961, 0.501961],
  [0.000000, 0.250980, 0.000000], [0.501961, 0.250980, 0.000000],
  [0.000000, 0.752941, 0.000000], [0.501961, 0.752941, 0.000000],
  [0.000000, 0.250980, 0.501961], [0.501961, 0.250980, 0.501961],
  [0.000000, 0.752941, 0.501961], [0.501961, 0.752941, 0.501961],
  [0.250980, 0.250980, 0.000000], [0.752941, 0.250980, 0.000000],
  [0.250980, 0.752941, 0.000000], [0.752941, 0.752941, 0.000000],
  [0.250980, 0.250980, 0.501961], [0.752941, 0.250980, 0.501961],
  [0.250980, 0.752941, 0.501961], [0.752941, 0.752941, 0.501961],
  [0.000000, 0.000000, 0.250980], [0.501961, 0.000000, 0.250980],
  [0.000000, 0.501961, 0.250980], [0.501961, 0.501961, 0.250980],
  [0.000000, 0.000000, 0.752941], [0.501961, 0.000000, 0.752941],
  [0.000000, 0.501961, 0.752941], [0.501961, 0.501961, 0.752941],
  [0.250980, 0.000000, 0.250980], [0.752941, 0.000000, 0.250980],
  [0.250980, 0.501961, 0.250980], [0.752941, 0.501961, 0.250980],
  [0.250980, 0.000000, 0.752941], [0.752941, 0.000000, 0.752941],
  [0.250980, 0.501961, 0.752941], [0.752941, 0.501961, 0.752941],
  [0.000000, 0.250980, 0.250980], [0.501961, 0.250980, 0.250980],
  [0.000000, 0.752941, 0.250980], [0.501961, 0.752941, 0.250980],
  [0.000000, 0.250980, 0.752941], [0.501961, 0.250980, 0.752941],
  [0.000000, 0.752941, 0.752941], [0.501961, 0.752941, 0.752941],
  [0.250980, 0.250980, 0.250980], [0.752941, 0.250980, 0.250980],
  [0.250980, 0.752941, 0.250980], [0.752941, 0.752941, 0.250980],
  [0.250980, 0.250980, 0.752941], [0.752941, 0.250980, 0.752941],
  [0.250980, 0.752941, 0.752941], [0.752941, 0.752941, 0.752941],
  [0.125490, 0.000000, 0.000000], [0.627451, 0.000000, 0.000000],
  [0.125490, 0.501961, 0.000000], [0.627451, 0.501961, 0.000000],
  [0.125490, 0.000000, 0.501961], [0.627451, 0.000000, 0.501961],
  [0.125490, 0.501961, 0.501961], [0.627451, 0.501961, 0.501961],
  [0.376471, 0.000000, 0.000000], [0.878431, 0.000000, 0.000000],
  [0.376471, 0.501961, 0.000000], [0.878431, 0.501961, 0.000000],
  [0.376471, 0.000000, 0.501961], [0.878431, 0.000000, 0.501961],
  [0.376471, 0.501961, 0.501961], [0.878431, 0.501961, 0.501961],
  [0.125490, 0.250980, 0.000000], [0.627451, 0.250980, 0.000000],
  [0.125490, 0.752941, 0.000000], [0.627451, 0.752941, 0.000000],
  [0.125490, 0.250980, 0.501961], [0.627451, 0.250980, 0.501961],
  [0.125490, 0.752941, 0.501961], [0.627451, 0.752941, 0.501961],
  [0.376471, 0.250980, 0.000000], [0.878431, 0.250980, 0.000000],
  [0.376471, 0.752941, 0.000000], [0.878431, 0.752941, 0.000000],
  [0.376471, 0.250980, 0.501961], [0.878431, 0.250980, 0.501961],
  [0.376471, 0.752941, 0.501961], [0.878431, 0.752941, 0.501961],
  [0.125490, 0.000000, 0.250980], [0.627451, 0.000000, 0.250980],
  [0.125490, 0.501961, 0.250980], [0.627451, 0.501961, 0.250980],
  [0.125490, 0.000000, 0.752941], [0.627451, 0.000000, 0.752941],
  [0.125490, 0.501961, 0.752941], [0.627451, 0.501961, 0.752941],
  [0.376471, 0.000000, 0.250980], [0.878431, 0.000000, 0.250980],
  [0.376471, 0.501961, 0.250980], [0.878431, 0.501961, 0.250980],
  [0.376471, 0.000000, 0.752941], [0.878431, 0.000000, 0.752941],
  [0.376471, 0.501961, 0.752941], [0.878431, 0.501961, 0.752941],
  [0.125490, 0.250980, 0.250980], [0.627451, 0.250980, 0.250980],
  [0.125490, 0.752941, 0.250980], [0.627451, 0.752941, 0.250980],
  [0.125490, 0.250980, 0.752941], [0.627451, 0.250980, 0.752941],
  [0.125490, 0.752941, 0.752941], [0.627451, 0.752941, 0.752941],
  [0.376471, 0.250980, 0.250980], [0.878431, 0.250980, 0.250980],
  [0.376471, 0.752941, 0.250980], [0.878431, 0.752941, 0.250980],
  [0.376471, 0.250980, 0.752941], [0.878431, 0.250980, 0.752941],
  [0.376471, 0.752941, 0.752941], [0.878431, 0.752941, 0.752941],
  [0.000000, 0.125490, 0.000000], [0.501961, 0.125490, 0.000000],
  [0.000000, 0.627451, 0.000000], [0.501961, 0.627451, 0.000000],
  [0.000000, 0.125490, 0.501961], [0.501961, 0.125490, 0.501961],
  [0.000000, 0.627451, 0.501961], [0.501961, 0.627451, 0.501961],
  [0.250980, 0.125490, 0.000000], [0.752941, 0.125490, 0.000000],
  [0.250980, 0.627451, 0.000000], [0.752941, 0.627451, 0.000000],
  [0.250980, 0.125490, 0.501961], [0.752941, 0.125490, 0.501961],
  [0.250980, 0.627451, 0.501961], [0.752941, 0.627451, 0.501961],
  [0.000000, 0.376471, 0.000000], [0.501961, 0.376471, 0.000000],
  [0.000000, 0.878431, 0.000000], [0.501961, 0.878431, 0.000000],
  [0.000000, 0.376471, 0.501961], [0.501961, 0.376471, 0.501961],
  [0.000000, 0.878431, 0.501961], [0.501961, 0.878431, 0.501961],
  [0.250980, 0.376471, 0.000000], [0.752941, 0.376471, 0.000000],
  [0.250980, 0.878431, 0.000000], [0.752941, 0.878431, 0.000000],
  [0.250980, 0.376471, 0.501961], [0.752941, 0.376471, 0.501961],
  [0.250980, 0.878431, 0.501961], [0.752941, 0.878431, 0.501961],
  [0.000000, 0.125490, 0.250980], [0.501961, 0.125490, 0.250980],
  [0.000000, 0.627451, 0.250980], [0.501961, 0.627451, 0.250980],
  [0.000000, 0.125490, 0.752941], [0.501961, 0.125490, 0.752941],
  [0.000000, 0.627451, 0.752941], [0.501961, 0.627451, 0.752941],
  [0.250980, 0.125490, 0.250980], [0.752941, 0.125490, 0.250980],
  [0.250980, 0.627451, 0.250980], [0.752941, 0.627451, 0.250980],
  [0.250980, 0.125490, 0.752941], [0.752941, 0.125490, 0.752941],
  [0.250980, 0.627451, 0.752941], [0.752941, 0.627451, 0.752941],
  [0.000000, 0.376471, 0.250980], [0.501961, 0.376471, 0.250980],
  [0.000000, 0.878431, 0.250980], [0.501961, 0.878431, 0.250980],
  [0.000000, 0.376471, 0.752941], [0.501961, 0.376471, 0.752941],
  [0.000000, 0.878431, 0.752941], [0.501961, 0.878431, 0.752941],
  [0.250980, 0.376471, 0.250980], [0.752941, 0.376471, 0.250980],
  [0.250980, 0.878431, 0.250980], [0.752941, 0.878431, 0.250980],
  [0.250980, 0.376471, 0.752941], [0.752941, 0.376471, 0.752941],
  [0.250980, 0.878431, 0.752941], [0.752941, 0.878431, 0.752941],
  [0.125490, 0.125490, 0.000000], [0.627451, 0.125490, 0.000000],
  [0.125490, 0.627451, 0.000000], [0.627451, 0.627451, 0.000000],
  [0.125490, 0.125490, 0.501961], [0.627451, 0.125490, 0.501961],
  [0.125490, 0.627451, 0.501961], [0.627451, 0.627451, 0.501961],
  [0.376471, 0.125490, 0.000000], [0.878431, 0.125490, 0.000000],
  [0.376471, 0.627451, 0.000000], [0.878431, 0.627451, 0.000000],
  [0.376471, 0.125490, 0.501961], [0.878431, 0.125490, 0.501961],
  [0.376471, 0.627451, 0.501961], [0.878431, 0.627451, 0.501961],
  [0.125490, 0.376471, 0.000000], [0.627451, 0.376471, 0.000000],
  [0.125490, 0.878431, 0.000000], [0.627451, 0.878431, 0.000000],
  [0.125490, 0.376471, 0.501961], [0.627451, 0.376471, 0.501961],
  [0.125490, 0.878431, 0.501961], [0.627451, 0.878431, 0.501961],
  [0.376471, 0.376471, 0.000000], [0.878431, 0.376471, 0.000000],
  [0.376471, 0.878431, 0.000000], [0.878431, 0.878431, 0.000000],
  [0.376471, 0.376471, 0.501961], [0.878431, 0.376471, 0.501961],
  [0.376471, 0.878431, 0.501961], [0.878431, 0.878431, 0.501961],
  [0.125490, 0.125490, 0.250980], [0.627451, 0.125490, 0.250980],
  [0.125490, 0.627451, 0.250980], [0.627451, 0.627451, 0.250980],
  [0.125490, 0.125490, 0.752941], [0.627451, 0.125490, 0.752941],
  [0.125490, 0.627451, 0.752941], [0.627451, 0.627451, 0.752941],
  [0.376471, 0.125490, 0.250980], [0.878431, 0.125490, 0.250980],
  [0.376471, 0.627451, 0.250980], [0.878431, 0.627451, 0.250980],
  [0.376471, 0.125490, 0.752941], [0.878431, 0.125490, 0.752941],
  [0.376471, 0.627451, 0.752941], [0.878431, 0.627451, 0.752941],
  [0.125490, 0.376471, 0.250980], [0.627451, 0.376471, 0.250980],
  [0.125490, 0.878431, 0.250980], [0.627451, 0.878431, 0.250980],
  [0.125490, 0.376471, 0.752941], [0.627451, 0.376471, 0.752941],
  [0.125490, 0.878431, 0.752941], [0.627451, 0.878431, 0.752941],
  [0.376471, 0.376471, 0.250980], [0.878431, 0.376471, 0.250980],
  [0.376471, 0.878431, 0.250980], [0.878431, 0.878431, 0.250980],
  [0.376471, 0.376471, 0.752941], [0.878431, 0.376471, 0.752941],
  [0.376471, 0.878431, 0.752941], [0.878431, 0.878431, 0.752941]
];

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})

export class AppComponent implements OnInit {
  title = 'interactive-visualizers';

  // Model related variables.
  modelDisplayName: string = DEFAULT_MODEL_DISPLAY_NAME;
  modelFormat: string|null = null; // Either 'tflite' or 'tfjs'.
  tfWebApiName: string|null = null;
  tfWebApi: any|null = null;
  modelMetadataUrl: string|null = null;
  modelMetadata: any|null = null;
  model: tf.GraphModel|null = null;
  labelmap: string[]|null = null;
  defaultScoreThreshold = 0.0;
  publisherName: string|null = null;
  publisherThumbnailUrl: string|null = null;

  // Query related variables.
  queryImageHeight: number|null = null;
  queryImageWidth: number|null = null;

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
  resultsLatency: number|null = null;
  // Classifier specific variables.
  classifierResults: Array<{displayName: string, score: number}>|null = null;
  // Detector specific variables.
  detectorResults: Array<{
    id: number,
    displayName: string,
    box: number[],
    score: number,
    label: number
  }>|null = null;
  detectionLabels: Array<{
    label: number,
    displayName: string,
    boxes: Array<{id: number, score: number}>,
    color: string
  }>|null = null;
  detectionScoreThreshold = DEFAULT_DETECTION_THRESHOLD;
  detectionLabelToIds = new Map();
  collapsedDetectionLabels = new Set();
  hoveredDetectionId: number|null = null;
  hoveredDetectionLabel: number|null = null;
  hoveredDetectionResultLabel: number|null = null;
  hoveredDetectionResultId: number|null = null;
  // Segmenter specific variables.
  segmenterPredictions: number[][]|null = null;
  segmenterLabelList: Array<{displayName: string, index: number,
    frequencyPercent: number, color: string}>|null = null;
  hoveredSegmentationLabel: number|null = null;

  ngOnInit(): void {
    // Sanity checks on URL query parameters.
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('testImagesIndexUrl')) {
      this.testImagesIndexUrl = urlParams.get('testImagesIndexUrl');
    }
    if (urlParams.has('modelDisplayName')) {
      this.modelDisplayName = urlParams.get('modelDisplayName');
    }
    if (urlParams.has('publisherName')) {
      this.publisherName = urlParams.get('publisherName');
    }
    if (urlParams.has('publisherThumbnailUrl')) {
      this.publisherThumbnailUrl = urlParams.get('publisherThumbnailUrl');
    }
    if (urlParams.has('tfliteModelUrl')) {
      if (!urlParams.has('tfWebApi')) {
        throw new Error(NO_TFWEB_API_ERROR_MESSAGE);
      }
      this.modelFormat = 'tflite';
      const tfliteModelUrl = decodeURIComponent(urlParams.get('tfliteModelUrl'));
      this.tfWebApiName = urlParams.get('tfWebApi');
      this.initAppWithTfliteModel(tfliteModelUrl);
    } else if (urlParams.has('modelMetadataUrl')) {
      this.modelFormat = 'tfjs';
      const modelMetadataUrl = decodeURIComponent(urlParams.get('modelMetadataUrl'));
      this.initAppWithTfjsModel(modelMetadataUrl);
    } else {
      throw new Error(NO_MODEL_ERROR_MESSAGE);
    }
  }

  async fetchTestImages(): Promise<void> {
    if (this.testImagesIndexUrl == null) {
      return;
    }
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

  async fetchModel(modelUrl: string): Promise<tf.GraphModel> {
    const model = await tf.loadGraphModel(modelUrl);
    return model;
  }

  /**
   * Initializes the app for the provided TFLite model URL.
   */
  async initAppWithTfliteModel(tfliteModelUrl: string): Promise<void> {
    switch (this.tfWebApiName) {
      case environment.imageClassifierApiName:
        this.tfWebApi = new ImageClassifierViz();
        break;
      case environment.imageSegmenterApiName:
        this.tfWebApi = new ImageSegmenterViz();
        break;
      case environment.objectDetectorApiName:
        this.tfWebApi = new ObjectDetectorViz();
        break;
    }
    await this.tfWebApi.init(environment.tfWebWasmFilesPrefix + this.tfWebApiName + '/', tfliteModelUrl);
    await this.warmUpModel();
    await this.fetchTestImages();
  }

  /**
   * Initializes the app for the provided model metadata URL.
   */
  async initAppWithTfjsModel(modelMetadataUrl: string): Promise<void> {
    // Load model & metadata.
    this.modelMetadataUrl = modelMetadataUrl;
    const metadataResponse = await fetch(this.modelMetadataUrl);
    this.modelMetadata = await metadataResponse.json();
    const modelUrl = this.getAssetsUrlPrefix() + 'model.json';
    this.model = await this.fetchModel(modelUrl);
    if (this.modelMetadata.tfjs_classifier_model_metadata) {
      this.tfWebApiName = environment.imageClassifierApiName;
    } else if (this.modelMetadata.tfjs_detector_model_metadata) {
      this.tfWebApiName = environment.objectDetectorApiName;
    } else if (this.modelMetadata.tfjs_segmenter_model_metadata) {
      this.tfWebApiName = environment.imageSegmenterApiName;
    }
    await this.fetchTestImages();
  }

  /**
   * Warms up the model. Subsequent calls will be faster.
   */
  async warmUpModel(): Promise<void> {
    return new Promise((resolve, reject) => {
      const warmUpImage = new Image();
      warmUpImage.onload = async () => {
        await this.tfWebApi.run(warmUpImage);
        resolve();
      };
      warmUpImage.onerror = () => reject();
      warmUpImage.src = WARMUP_IMAGE_DATA_URL;
    });
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
      switch (this.tfWebApiName) {
        case environment.imageClassifierApiName:
          this.runImageClassifier(image, index);
          break;
        case environment.objectDetectorApiName:
          this.runObjectDetector(image, index);
          break;
        case environment.imageSegmenterApiName:
          this.runImageSegmenter(image, index);
          break;
        default:
          console.error(
              `The TFWeb API \`${this.tfWebApi}\ isn't currently supported.`);
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
   * Run the model in case of image classification.
   */
  async runImageClassifier(image: HTMLImageElement, index: number):
      Promise<void> {
    let results = [];
    if (this.modelFormat === 'tflite') {
      const startTs = Date.now();
      const rawResults = this.tfWebApi.run(image);
      this.resultsLatency = Date.now() - startTs;
      rawResults.getClassificationsList()[0].getClassesList().forEach(cls => {
        if (cls.getDisplayName()) {
          results.push({
            displayName: cls.getDisplayName(),
            score: cls.getScore(),
          });
        } else {
          results.push({
            displayName: cls.getClassName(),
            score: cls.getScore(),
          });
        }
      });
    } else {
      // Prepare inputs.
      const inputTensorMetadata =
          this.modelMetadata.tfjs_classifier_model_metadata.input_tensor_metadata;
      const imageTensor = this.prepareImageInput(image, inputTensorMetadata);

      // Execute the model.
      const startTs = Date.now();
      const outputTensor: tf.Tensor =
          await this.model.executeAsync(imageTensor) as tf.Tensor;
      this.resultsLatency = Date.now() - startTs;
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

    if (this.imageSelectedIndex === index) {
      // Display results only for the last selected image (as the user may
      // have switched selection while inference was running).
      this.classifierResults = results;
      this.resultsKeyName = 'Type';
      this.resultsValueName = 'Score';
    }
  }

  /**
   * Run the model in case of image segmentation.
   */
  async runImageSegmenter(image: HTMLImageElement, index: number):
      Promise<void> {
    let predictions: number[][] = [];
    if (this.modelFormat === 'tflite') {
      const startTs = Date.now();
      const segmentation = this.tfWebApi.run(image).getSegmentationList()[0];
      this.resultsLatency = Date.now() - startTs;
      const categoryMask = segmentation.getCategoryMask();
      for (let i = 0; i < segmentation.getHeight(); i++) {
        predictions.push(Array.from(categoryMask.slice(segmentation.getWidth() * i,
          segmentation.getWidth() * (i + 1))));
      }
      this.labelmap = [];
      const coloredLabelList = segmentation.getColoredLabelsList();
      for (let i = 0; i < coloredLabelList.length; i++) {
        const colorLabel = coloredLabelList[i];
        if (colorLabel.getDisplayName()) {
          this.labelmap.push(colorLabel.getDisplayName());
        } else {
          this.labelmap.push(colorLabel.getClassName());
        }
        if (colorLabel.getR() && colorLabel.getG() && colorLabel.getB()) {
          COLOR_LIST[i] = [colorLabel.getR() / 255, colorLabel.getG() / 255, colorLabel.getB() / 255];
        }
      }
    } else {
      // Prepare inputs.
      const inputTensorMetadata =
          this.modelMetadata.tfjs_segmenter_model_metadata.input_tensor_metadata;
      const imageTensor = this.prepareImageInput(image, inputTensorMetadata);

      // Execute the model.
      const outputHeadMetadata =
          this.modelMetadata.tfjs_segmenter_model_metadata.output_head_metadata[0];
      const outputTensorName =
          outputHeadMetadata.semantic_predictions_tensor_name;
      const startTs = Date.now();
      const outputTensor =
          await this.model.executeAsync(imageTensor, outputTensorName) as tf.Tensor;
      this.resultsLatency = Date.now() - startTs;
      tf.dispose(imageTensor);
      const squeezedOutputTensor = outputTensor.squeeze();
      tf.dispose(outputTensor);
      predictions = await squeezedOutputTensor.array() as number[][];
      tf.dispose(squeezedOutputTensor);

      // Fetch the labelmap.
      if (this.labelmap == null && outputHeadMetadata.labelmap_path != null) {
        await this.fetchLabelmap(outputHeadMetadata.labelmap_path);
      }
    }

    // Generate labelmap if not found.
    if (this.labelmap == null) {
      let maxLabelIndex = 0;
      for (const predictionLine of predictions) {
        for (const prediction of predictionLine) {
          maxLabelIndex = Math.max(maxLabelIndex, prediction);
        }
      }
      this.labelmap = [];
      for (let i = 0; i <= maxLabelIndex; ++i) {
        this.labelmap.push(`Label ${i}`);
      }
    }

    // Compute label frequencies.
    const frequencies = new Array(this.labelmap.length).fill(0);
    for (const predictionLine of predictions) {
        for (const prediction of predictionLine) {
        ++frequencies[prediction];
      }
    }

    // Sort labels by decreasing area importance in the query image.
    const labelList = frequencies
                          .map((frequency, listIndex) => {
                            return {
                              displayName: this.labelmap[listIndex],
                              index: listIndex,
                              frequencyPercent: Math.ceil(
                                  100 * frequency /
                                  (predictions.length * predictions[0].length)),
                              color: `rgb(${255 * COLOR_LIST[listIndex][0]}, ${255 *
          COLOR_LIST[listIndex][1]}, ${255 * COLOR_LIST[listIndex][2]})`,
                            };
                          })
                          .filter(x => x.frequencyPercent > EPSILON)
                          .sort((a, b) => {
                            if (a.frequencyPercent > b.frequencyPercent) {
                              return -1;
                            }
                            return 1;
                          });

    if (this.imageSelectedIndex === index) {
      // Display results only for the last selected image (as the user may
      // have switched selection while inference was running).
      this.segmenterPredictions = predictions;
      this.segmenterLabelList = labelList;
      this.hoveredSegmentationLabel = null;
      this.resultsKeyName = 'Type';
      this.resultsValueName = 'Percentage of image area';

      const imageHtmlElement = document.getElementById('query-image') as HTMLImageElement;
      this.queryImageHeight = imageHtmlElement.offsetHeight;
      this.queryImageWidth = imageHtmlElement.offsetWidth;
      const width = predictions.length;
      const height = predictions[0].length;
      const canvas = document.getElementById('query-canvas-overlay') as HTMLCanvasElement;
      canvas.style.height = `${this.queryImageHeight}px`;
      canvas.style.width = `${this.queryImageWidth}px`;
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d') as CanvasRenderingContext2D;
      context.fillRect(0, 0, width, height);
      this.fillSegmentationCanvas();

      canvas.addEventListener(
          'mousemove', (event => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = width / rect.width;
            const scaleY = height / rect.height;
            const x = Math.min(
                width - 1,
                Math.max(0, Math.round((event.clientX - rect.left) * scaleX)));
            const y = Math.min(
                height - 1,
                Math.max(0, Math.round((event.clientY - rect.top) * scaleY)));
            const hoveredLabel = this.segmenterPredictions[y][x];
            if (hoveredLabel !== this.hoveredSegmentationLabel) {
              this.hoveredSegmentationLabel = hoveredLabel;
              this.fillSegmentationCanvas();
            }
          }));

      canvas.addEventListener('mouseout', (event => {
                                this.hoveredSegmentationLabel = null;
                                this.fillSegmentationCanvas();
                              }));
    }
  }

  /**
   * Fills a canvas with segmenter predictions overlaid on top of the query image.
   */
  fillSegmentationCanvas(): void {
    const canvas = document.getElementById('query-canvas-overlay') as HTMLCanvasElement;
    canvas.style.cursor = 'pointer';
    const context = canvas.getContext('2d') as CanvasRenderingContext2D;
    const height = this.segmenterPredictions.length;
    const width = this.segmenterPredictions[0].length;
    const imageData = context.getImageData(0, 0, width, height);
    const data = imageData.data;
    for (let i = 0; i < height; ++i) {
      for (let j = 0; j < width; ++j) {
        const labelIndex = this.segmenterPredictions[i][j];
        const currentPixel = 4 * (width * i + j);
        if (this.hoveredSegmentationLabel != null &&
            labelIndex !== this.hoveredSegmentationLabel) {
          data[currentPixel] = 0;
          data[currentPixel + 1] = 0;
          data[currentPixel + 2] = 0;
          data[currentPixel + 3] = 0;  // Fully transparent.
        } else {
          data[currentPixel] = 255 * COLOR_LIST[labelIndex][0];
          data[currentPixel + 1] = 255 * COLOR_LIST[labelIndex][1];
          data[currentPixel + 2] = 255 * COLOR_LIST[labelIndex][2];
          data[currentPixel + 3] = 150;
        }
      }
    }
    context.putImageData(imageData, 0, 0);
  }

  segmenterResultHovered(index: number): void {
    this.hoveredSegmentationLabel = index;
    this.fillSegmentationCanvas();
  }

  segmenterResultLeft(): void {
    this.hoveredSegmentationLabel = null;
    this.fillSegmentationCanvas();
  }

  /**
   * Run the model in case of image detection, and return detector results.
   */
  async runObjectDetector(image: HTMLImageElement, index: number):
      Promise<void> {
    const results = [];
    if (this.modelFormat === 'tflite') {
      const startTs = Date.now();
      const detections = this.tfWebApi.run(image).getDetectionsList();
      this.resultsLatency = Date.now() - startTs;
      for (let i = 0; i < detections.length; i++) {
        const detection = detections[i];
        const boundingBox = detection.getBoundingBox();
        const top = boundingBox.getOriginY() / image.height;
        const left = boundingBox.getOriginX() / image.width;
        const bottom = (boundingBox.getOriginY() + boundingBox.getHeight()) / image.height;
        const right = (boundingBox.getOriginX() + boundingBox.getWidth()) / image.width;
        let displayName = detection.getClassesList()[0].getDisplayName();
        if (!displayName) {
          displayName = detection.getClassesList()[0].getClassName();
        }
        results.push({
          id: i,
          box: [top, left, bottom, right],
          score: detection.getClassesList()[0].getScore(),
          label: detection.getClassesList()[0].getClassName(),
          displayName,
        });
      }
    } else {
      // Prepare inputs.
      const inputTensorMetadata =
          this.modelMetadata.tfjs_detector_model_metadata.input_tensor_metadata;
      const imageTensor = this.prepareImageInput(image, inputTensorMetadata);

      // Execute the model.
      const outputHeadMetadata =
          this.modelMetadata.tfjs_detector_model_metadata.output_head_metadata[0];
      const numDetectionsTensorName =
          outputHeadMetadata.num_detections_tensor_name;
      const detectionBoxesTensorName =
          outputHeadMetadata.detection_boxes_tensor_name;
      const detectionScoresTensorName =
          outputHeadMetadata.detection_scores_tensor_name;
      const detectionClassesTensorName =
          outputHeadMetadata.detection_classes_tensor_name;
      const startTs = Date.now();
      const outputTensors = await this.model.executeAsync(imageTensor, [
        numDetectionsTensorName, detectionBoxesTensorName,
        detectionScoresTensorName, detectionClassesTensorName
      ]) as tf.Tensor[];
      this.resultsLatency = Date.now() - startTs;
      tf.dispose(imageTensor);
      const squeezedNumDetections = await outputTensors[0].squeeze();
      const squeezedDetectionBoxes = await outputTensors[1].squeeze();
      const squeezedDetectionScores = await outputTensors[2].squeeze();
      const squeezedDetectionClasses = await outputTensors[3].squeeze();
      tf.dispose(outputTensors);
      const numDetections = await squeezedNumDetections.array() as number;
      const detectionBoxes = await squeezedDetectionBoxes.array() as number[][];
      const detectionScores = await squeezedDetectionScores.array() as number[];
      const detectionClasses = await squeezedDetectionClasses.array() as number[];
      tf.dispose(squeezedNumDetections);
      tf.dispose(squeezedDetectionBoxes);
      tf.dispose(squeezedDetectionScores);
      tf.dispose(squeezedDetectionClasses);

      // Fetch labelmap and score thresholds.
      this.detectionScoreThreshold = DEFAULT_DETECTION_THRESHOLD;
      if (outputHeadMetadata.score_threshold != null &&
          outputHeadMetadata.score_threshold.length) {
        this.detectionScoreThreshold = outputHeadMetadata.score_threshold[0];
      }
      if (this.labelmap == null && outputHeadMetadata.labelmap_path != null) {
        await this.fetchLabelmap(outputHeadMetadata.labelmap_path);
      }

      for (let i = 0; i < numDetections; ++i) {
        const label = detectionClasses[i];
        if (this.labelmap != null && this.labelmap.length > label) {
          results.push({
            id: i,
            box: detectionBoxes[i],
            score: detectionScores[i],
            label,
            displayName: this.labelmap[label],
          });
        } else {
          results.push({
            id: i,
            box: detectionBoxes[i],
            score: detectionScores[i],
            label,
            displayName: label,
          });
        }
      }
    }

    // Display results only for the last selected image (as the user may
    // have switched selection while inference was running).
    if (this.imageSelectedIndex !== index) {
      return;
    }

    // Prepare detector results general variables.
    this.detectorResults = results;
    this.resultsKeyName = 'Type';
    this.resultsValueName = 'Score';
    this.detectionLabelToIds.clear();
    for (const result of results) {
      if (!this.detectionLabelToIds.has(result.label)) {
        const newId = this.detectionLabelToIds.size;
        this.detectionLabelToIds.set(result.label, newId);
      }
    }
    this.hoveredDetectionResultLabel = null;
    this.hoveredDetectionResultId = null;
    this.hoveredDetectionLabel = null;
    this.hoveredDetectionId = null;
    this.collapsedDetectionLabels = new Set();

    // Prepare the result list content.
    const labelDisplayNames = new Map();
    const labelBoxes = new Map();
    for (const result of results) {
      if (!labelDisplayNames.has(result.label)) {
        labelDisplayNames.set(result.label, result.displayName);
        labelBoxes.set(result.label, []);
      }
      if (!this.collapsedDetectionLabels.has(result.label)) {
        labelBoxes.get(result.label).push({
          id: result.id,
          score: result.score,
          rect: result.box,
        });
      }
    }
    const labels = [];
    for (const label of Array.from(labelDisplayNames.keys())) {
      const colorIndex = this.detectionLabelToIds.get(label);
      labels.push({
        label,
        displayName: labelDisplayNames.get(label),
        boxes: labelBoxes.get(label),
        color: `rgb(${255 * COLOR_LIST[colorIndex][0]}, ${
            255 *
            COLOR_LIST[colorIndex][1]}, ${255 * COLOR_LIST[colorIndex][2]})`,

      });
    }
    this.detectionLabels = labels;

    const imageHtmlElement = document.getElementById('query-image') as HTMLImageElement;
    this.queryImageHeight = imageHtmlElement.offsetHeight;
    this.queryImageWidth = imageHtmlElement.offsetWidth;

    // Update score threshold position after letting time for the UI to update.
    setTimeout(() => this.updateDetectionScoreThresholdPosition(), 20);
  }

  /**
   * Coordinates are expected to be in the [0, 1] range.
   */
  displayRectangles(detections: Array<{id: number, displayName: string, box: number[], score: number, label: number}>): void {
    const canvas = document.getElementById('query-canvas-overlay') as HTMLCanvasElement;
    canvas.style.height = `${this.queryImageHeight}px`;
    canvas.style.width = `${this.queryImageWidth}px`;
    canvas.width = this.queryImageWidth;
    canvas.height = this.queryImageHeight;
    const context = canvas.getContext('2d') as CanvasRenderingContext2D;
    context.globalAlpha = 0.5;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.globalAlpha = 1;
    context.fillStyle = 'white';
    context.lineWidth = DETECTION_RECTANGLE_BORDER_WIDTH;

    // First remove rectangle contents.
    context.globalCompositeOperation = 'destination-out';
    for (const detection of detections) {
      const rectangle = detection.box;
      const top = rectangle[0];
      const left = rectangle[1];
      const bottom = rectangle[2];
      const right = rectangle[3];
      const width = right - left;
      const height = bottom - top;
      context.fillRect(
          this.queryImageWidth * left, this.queryImageHeight * top, this.queryImageWidth * width,
          this.queryImageHeight * height);
    }

    // Then draw the borders.
    context.globalCompositeOperation = 'source-over';
    for (const detection of detections) {
      const rectangle = detection.box;
      const top = rectangle[0];
      const left = rectangle[1];
      const bottom = rectangle[2];
      const right = rectangle[3];
      const width = right - left;
      const height = bottom - top;
      const colorIndex = this.detectionLabelToIds.get(detection.label);
      context.strokeStyle = `rgb(${255 * COLOR_LIST[colorIndex][0]}, ${255 *
          COLOR_LIST[colorIndex][1]}, ${255 * COLOR_LIST[colorIndex][2]})`;
      context.beginPath();
      context.setLineDash([3, 3]);
      context.moveTo(this.queryImageWidth * left, this.queryImageHeight * top);
      context.lineTo(this.queryImageWidth * (left + width), this.queryImageHeight * top);
      context.lineTo(this.queryImageWidth * (left + width), this.queryImageHeight * (top + height));
      context.lineTo(this.queryImageWidth * left, this.queryImageHeight * (top + height));
      context.lineTo(this.queryImageWidth * left, this.queryImageHeight * top);
      context.stroke();
    }
  }

  /*
   * Removes the query image overlaid canvas content.
   */
  removeOverlayedCanvas(): void {
    const canvas = document.getElementById('query-canvas-overlay') as HTMLCanvasElement;
    canvas.width = 0;
    canvas.height = 0;
    const context = canvas.getContext('2d') as CanvasRenderingContext2D;
    context.fillRect(0, 0, 0, 0);
  }

  /**
   * Fills the query image overlaid canvas with detection results.
   */
  fillDetectionCanvas(): void {
    if (this.detectorResults == null) {
      return;
    }
    this.removeOverlayedCanvas();
    document.getElementById('query-image').style.opacity = '1';

    if (this.hoveredDetectionResultLabel != null) {
      // Case a result label is hovered. Displays the bounding boxes it's part of,
      // and ignore other detections.
      const detections = [];
      for (const detection of this.detectorResults) {
        if (detection.label === this.hoveredDetectionResultLabel &&
            detection.score >= this.detectionScoreThreshold) {
          detections.push(detection);
        }
      }
      this.displayRectangles(detections);
    } else if (this.hoveredDetectionResultId != null) {
      // Case a result box ID is hovered. Displays only its boulding box.
      const detections = [];
      for (const detection of this.detectorResults) {
        if (detection.id === this.hoveredDetectionResultId &&
            detection.score >= this.detectionScoreThreshold) {
          detections.push(detection);
        }
      }
      this.displayRectangles(detections);
    }
  }

  /** On click on a detector label. */
  detectorResultLabelClicked(label: number): void {
    if (this.collapsedDetectionLabels.has(label)) {
      this.collapsedDetectionLabels.delete(label);
    } else {
      this.collapsedDetectionLabels.add(label);
    }
  }

  /** On hover on a detector label. */
  detectorResultLabelHovered(label: number): void {
    this.hoveredDetectionResultLabel = label;
    this.fillDetectionCanvas();
  }

  /** When leaving a detector result or label hover. */
  canvasOverlayLeft(): void {
    if (this.detectorResults != null) {
      this.hoveredDetectionResultLabel = null;
      this.hoveredDetectionResultId = null;
      this.removeOverlayedCanvas();
    }
  }

  /** On hover on a specific detector result. */
  detectorResultIdHovered(resultId: number): void {
    this.hoveredDetectionResultId = resultId;
    this.fillDetectionCanvas();
  }

  /** On click on a hint overlayd on the query image. */
  detectorHintClicked(resultId: number): void {
    this.hoveredDetectionResultId = resultId;
    this.fillDetectionCanvas();
  }

  detectionScoreThresholdChanged(event: InputEvent): void {
    const sliderElement = event.target as HTMLInputElement;
    this.detectionScoreThreshold = parseFloat(sliderElement.value) / 100;

    // Update score threshold position after letting time for the UI to update.
    setTimeout(() => this.updateDetectionScoreThresholdPosition(), 20);
  }

  updateDetectionScoreThresholdPosition(): void {
    const thresholdSliderValueElement =
        document.getElementById('threshold-slider-value');
    const width = thresholdSliderValueElement.getBoundingClientRect().width;
    thresholdSliderValueElement.style.marginLeft =
      `calc(13px + ${this.detectionScoreThreshold} * (100% - 42px) - ${width}px / 2)`;
  }

  /**
   * Copies the embed URL to the clipboard.
   */
  copyEmbedUrl(): void {
    // Build a temporary textarea element with config URL to be copied to
    // clipboard.
    const element = document.createElement('textarea') as HTMLTextAreaElement;
    element.value = window.location.href;
    document.body.appendChild(element);
    element.select();
    // Copy the URL to clipboard.
    document.execCommand('copy');
    // Remove the temporary textarea element.
    document.body.removeChild(element);
    alert('Embed URL copied to clipboard.');
  }
}
