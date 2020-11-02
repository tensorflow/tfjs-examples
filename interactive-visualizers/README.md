# TensorFlow.js Interactive Model Visualizers

TensorFlow.js-powered Interactive Model Visualizers for standard perception
tasks embeddable anywhere in the web.

Supported tasks:

*   Image Classification
*   Object Detection
*   Image Segmentation

The Interactive Visualizer supports any model coming with a metadata JSON file
formatted following the supported tasks standards. This metadata file is passed
at runtime to the visualizer as URL query parameter (e.g.
`https://visualizerHostedUrl/?modelMetadataUrl=https://myModelMetadataUrl/metadata.json`).

The visualizer will then load its UI and behave according to the task defined by
the model metadata.

NOTE: standards definition is WIP. Here's an example of such metadata for an
image classifier model:
https://storage.googleapis.com/tfhub-visualizers/google/aiy/vision/classifier/plants_V1/1/metadata.json.

This project was generated with
[Angular CLI](https://github.com/angular/angular-cli) version 10.1.3.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app
will automatically reload if you change any of the source files.

## Build

Run `ng build` to build the project. The build artifacts will be stored in the
`dist/` directory. Use the `--prod` flag for a production build.

## Test

Run `ng test` to run the tests.
[Karma](https://karma-runner.github.io/latest/index.html) is providing the
testing environment, and [Jasmine](https://jasmine.github.io/) is used for unit
testing.

Additionally run `ng lint` to check for the linter warnings/errors.
