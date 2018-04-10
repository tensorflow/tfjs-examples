# TensorFlow.js Examples

This repository contains a set of examples implemented in
[TensorFlow.js](http://js.tensorflow.org).

Each example directory is standalone so the directory can be copied
to another project.

## How to build an example
`cd` into the directory, run `yarn` to install
the dependencies of the example, and `yarn watch` to build it. For example:

```sh
cd mnist-core
yarn
yarn watch
```
If you use `npm`, you can:
```sh
cd mnist-core
npm install
npm run watch
```

### Details

The convention is that each example contains two scripts:

- `yarn watch` or `npm run watch`: starts a local development HTTP server which watches the
filesystem for changes so you can edit the code (JS or HTML) and see changes when you refresh the page immediately.

- `yarn build` or `npm run build`: generates a `dist/` folder which contains the build artifacts and
can be used for deployment.

## Contributing

If you want to contribute an example, please reach out to us on
[Github issues](https://github.com/tensorflow/tfjs-examples/issues)
before sending us a pull request as we are trying to keep this set of examples
small and highly curated.
