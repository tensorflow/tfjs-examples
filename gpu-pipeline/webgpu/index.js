/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

async function init() {
  const customBackendName = 'custom-webgpu';

  const kernels = tf.getKernelsForBackend('webgpu');
  kernels.forEach(kernelConfig => {
    const newKernelConfig = { ...kernelConfig, backendName: customBackendName };
    tf.registerKernel(newKernelConfig);
  });

  adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();

  tf.registerBackend(customBackendName, async () => {
    return new tf.WebGPUBackend(device);
  });
  await tf.setBackend(customBackendName);

  const context = canvasEl.getContext('webgpu');
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  const presentationSize = [
    canvasEl.width,
    canvasEl.height,
  ];

  context.configure({
    device,
    size: presentationSize,
    format: presentationFormat,
    alphaMode: 'opaque',
  });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({
        code: VERTEX_SHADER,
      }),
      entryPoint: 'main',
    },
    fragment: {
      module: device.createShaderModule({
        code: PIXEL_SHADER,
      }),
      entryPoint: 'main',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const sizeParams = {
    width: canvasEl.width,
    height: canvasEl.height,
  };

  const sizeParamBuffer = device.createBuffer({
    size: 2 * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(sizeParamBuffer, 0, new Int32Array([sizeParams.width, sizeParams.height]));

  const predict = async () => {
    beginEstimateSegmentationStats();
    const segmentationConfig = {flipHorizontal: false, multiSegmentation: false, segmentBodyParts: true,
      segmentationThreshold: 0.5};
    const segmentation = await model.segmentPeople(video, segmentationConfig);

    const tensor = await segmentation[0].mask.toTensor();
    const data = tensor.dataToGPU();

    const uniformBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 1,
          resource: sampler,
        },
        {
          binding: 2,
          resource: device.importExternalTexture({
            source: video,
          }),
        },
        {
          binding: 3,
          resource: {
            buffer: data.buffer,
            size: data.bufSize,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: sizeParamBuffer,
          },
        }
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    tensor.dispose();
    data.tensorRef.dispose();
    endEstimateSegmentationStats();

    requestAnimationFrame(predict);
  }
  requestAnimationFrame(predict);
}

async function start() {
  await tf.ready();
  setupPage();
}

start();
