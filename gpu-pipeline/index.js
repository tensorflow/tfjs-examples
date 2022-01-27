let video, model, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;

const imageEl = document.querySelector("img");
const statusEl = document.querySelector("#status");
const canvasEl = document.querySelector("canvas");
const videoWidth = isMobile() ? 480 : 640;
const videoHeight = isMobile() ? 480 : 480;
canvasEl.width = videoWidth;
canvasEl.height = videoHeight;
canvasEl.style.width = `${videoWidth}px`;
canvasEl.style.height = `${videoHeight}px`;

async function init() {
  if (tf.findBackendFactory('webgl')) {
    tf.removeBackend('webgl');
  }
  // Must be named 'webgl' to match the kernel namespace.
  const gl = getWebGLRenderingContext(canvasEl);
  tf.registerBackend('webgl', () => {
    return new tf.MathBackendWebGL(
        new tf.GPGPUContext(gl));
  });
  await tf.setBackend('webgl');

  const applyMask = new MaskStep(gl);
  let inputTextureFrameBuffer = createTextureFrameBuffer(gl, gl.LINEAR, videoWidth, videoHeight);

  const predict = async () => {
    beginEstimateSegmentationStats();
      
    // Put original video content on the input texture.
    inputTextureFrameBuffer.bindTexture();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, video);

    // Run segmentation model inference.
    const segmentationConfig = {flipHorizontal: false, multiSegmentation: false, segmentBodyParts: true,
      segmentationThreshold: 0.5};
    segmentation = await model.segmentPeople(video, segmentationConfig);

    // Get the tensor result and the texture that holds the data.
    // We tell the system to use the video width and height as the tex shape,
    // this allows the densely packed data to have the same layout as the 
    // original video content, which simplifies the shader logic. This only
    // works if the data shape is [1, height, width, 4].
    const tensor = await segmentation[0].mask.toTensor();
    const data =
    tensor.dataToGPU({customTexShape: [videoHeight, videoWidth]});

    // Combine the input texture and tensor texture with additional shader logic.
    // In this case, we just pass through foreground pixels and make background
    // pixels more transparent.
    result = applyMask.process(inputTextureFrameBuffer, createTexture(
        gl, data.texture, videoWidth, videoHeight));
    
    // Other processing steps can go here.

    // Once we're done with all the processing, we can draw the texture on the canvas.
    
    // Making gl.DRAW_FRAMEBUFFER to be null sets rendering back to default framebuffer.
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    // Caching the data of the result texture to be drawn in the gl.READ_FRAMEBUFFER.
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, result.framebuffer_);
    // Transfer the data from read framebuffer to the default framebuffer to make it show
    // on canvas.
    gl.blitFramebuffer(
        0, 0, videoWidth, videoHeight, 0, videoHeight, videoWidth, 0, gl.COLOR_BUFFER_BIT,
        gl.LINEAR);

    // Make sure to dispose all tensors, otherwise there will be memory leak.
    tensor.dispose();
    data.tensorRef.dispose();
    endEstimateSegmentationStats();

    requestAnimationFrame(predict);
  };

  predict();
}

setupPage();
