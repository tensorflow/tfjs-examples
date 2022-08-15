# Overview

A quick demo for running [TFJS mobilenet model][mobilenet] using
[TFJS React Native][tfjs-react-native] with Expo and react native CLI.

## Expo

```
$ cd expo
$ yarn
$ yarn start
```

Then scan the QR code to open it in the `Expo Go` app.

## React native CLI

```bash
$ cd react-native-cli
$ yarn
# Start Metro
$ yarn start --reset-cache
# Run on android (only android is configured for the demo).
$ yarn android
```

Tested on iPhone 12 Pro Max and Pixel 2.

<img src="screenshot.jpg" width="300">

[mobilenet]: https://github.com/tensorflow/tfjs-models/tree/master/mobilenet
[tfjs-react-native]: https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native
