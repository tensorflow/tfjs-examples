import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, Text, View, Dimensions, Platform } from 'react-native';

import { Camera } from 'expo-camera';

import * as tf from '@tensorflow/tfjs';
import * as posedetection from '@tensorflow-models/pose-detection';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import Svg, { Circle } from 'react-native-svg';
import { ExpoWebGLRenderingContext } from 'expo-gl';

// tslint:disable-next-line: variable-name
const TensorCamera = cameraWithTensors(Camera);

// Camera preview size.
//
// From experiments, to render camera feed without distortion, 16:9 ratio
// should be used fo iOS devices and 4:3 ratio should be used for android
// devices.
//
// This might not cover all cases.
const CAM_PREVIEW_WIDTH = Dimensions.get('window').width;
const CAM_PREVIEW_HEIGHT =
  CAM_PREVIEW_WIDTH / (Platform.OS === 'ios' ? 9 / 16 : 3 / 4);

// The score threshold for pose detection results.
const MIN_KEYPOINT_SCORE = 0.3;

// The size of the resized output from TensorCamera.
//
// For movenet, the size here doesn't matter too much because the model will
// preprocess the input (crop, resize, etc).
const OUTPUT_TENSOR_WIDTH = 240;
const OUTPUT_TENSOR_HEIGHT = 320;

// Whether to auto-render TensorCamera preview.
const AUTO_RENDER = false;

export default function App() {
  const cameraRef = useRef(null);
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState<posedetection.PoseDetector>();
  const [poses, setPoses] = useState<posedetection.Pose[]>();
  const [fps, setFps] = useState(0);

  useEffect(() => {
    async function prepare() {
      // Camera permission.
      await Camera.requestPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      // Load movenet model.
      // https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
      const model = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        {
          modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        }
      );
      setModel(model);

      // Ready!
      setTfReady(true);
    }

    prepare();
  }, []);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      // Get the tensor and run pose detection.
      const imageTensor = images.next().value as tf.Tensor3D;

      const startTs = Date.now();
      const poses = await model!.estimatePoses(
        imageTensor,
        undefined,
        Date.now()
      );
      const latency = Date.now() - startTs;
      setFps(Math.floor(1000 / latency));
      setPoses(poses);
      tf.dispose([imageTensor]);

      // Render camera preview manually when autorender=false.
      if (!AUTO_RENDER) {
        updatePreview();
        gl.endFrameEXP();
      }

      requestAnimationFrame(loop);
    };

    loop();
  };

  const renderPose = () => {
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints
        .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
        .map((k) => {
          // Flip horizontally on android.
          const x = Platform.OS === 'android' ? OUTPUT_TENSOR_WIDTH - k.x : k.x;
          const y = k.y;
          return (
            <Circle
              key={`skeletonkp_${k.name}`}
              cx={(x / OUTPUT_TENSOR_WIDTH) * CAM_PREVIEW_WIDTH}
              cy={(y / OUTPUT_TENSOR_HEIGHT) * CAM_PREVIEW_HEIGHT}
              r='4'
              strokeWidth='2'
              fill='#00AA00'
              stroke='white'
            />
          );
        });

      return <Svg style={styles.svg}>{keypoints}</Svg>;
    } else {
      return <View></View>;
    }
  };

  const renderFps = () => {
    return (
      <View style={styles.fpsContainer}>
        <Text>FPS: {fps}</Text>
      </View>
    );
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    return (
      // Note that you don't need to specify `cameraTextureWidth` and
      // `cameraTextureHeight` prop in `TensorCamera` below.
      <View style={styles.container}>
        <TensorCamera
          ref={cameraRef}
          style={styles.camera}
          autorender={AUTO_RENDER}
          type={Camera.Constants.Type.front}
          // tensor related props
          resizeWidth={OUTPUT_TENSOR_WIDTH}
          resizeHeight={OUTPUT_TENSOR_HEIGHT}
          resizeDepth={3}
          onReady={handleCameraStream}
        />
        {renderPose()}
        {renderFps()}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  loadingMsg: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    position: 'absolute',
    top: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
    left: Dimensions.get('window').width / 2 - CAM_PREVIEW_WIDTH / 2,
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    zIndex: 1,
  },
  svg: {
    top: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
    left: Dimensions.get('window').width / 2 - CAM_PREVIEW_WIDTH / 2,
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    position: 'absolute',
    zIndex: 30,
  },
  fpsContainer: {
    position: 'absolute',
    top: 80,
    left: 10,
    width: 80,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
});
