# TensorFlow.js Example: React Native

This directory hosts a set of react native apps that use
[`tfjs-react-native`](https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native)
package. Please refer to their own README files for more info.

If the demo app crashes on startup, it is highly likely caused by incompatible
package versions, specifically `expo-gl` and `react-native`. As of Jan 2022,
the following version combination should work. It is tested on iPhone 13 Pro
Max with iOS 15.1.1 and Pixel 2 with Android 9:

```
"expo": "~44.0.2",
"expo-camera": "^12.1.0",
"expo-file-system": "^13.2.0",
"expo-gl": "^11.1.1",
"expo-gl-cpp": "^11.1.0",
"react": "17.0.1",
"react-native": "~0.64.3",
```
