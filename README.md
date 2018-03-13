# TensorFlow.js Examples

This repository contains a number of different examples implemented in
[TensorFlow.js](http://js.tensorflow.org).

Each example directory is completely standalone so you can copy the directory
to another project.

## How to build an example

To build an example, first `cd` into the directory and run `yarn` to install
the dependencies of the example, for example:

```
cd mnist-core
yarn
```

The convention is that each example contains two scripts:

- `yarn watch`: starts a local development HTTP server which watches the
filesystem for changes so you can edit the code (JS or HTML) and immediately
see changes when you refresh the page.

- `yarn build`: generates a `dist/` folder which contains the build artifacts and
can be used for deployment.

## Contributing

If you want to contribute an example, please reach out to us on Github issues
before sending us a pull request as we are trying to keep this set of examples
small and highly curated.
