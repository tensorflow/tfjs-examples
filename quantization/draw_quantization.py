# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Draws diargams to provide an intuitive understanding of the discretization
# that result from 16-bit and 8-bit weight quantization.

from matplotlib import pyplot as plt
import numpy as np


def quantize(w, bits):
  """
  Simulate weight quantization.

  Args:
    w: (a numpy.ndarray) The weight to be quantized.
    bits: (int) number of bits used for the quantization: 8 or 16.

  Returns:
    A tuple with three elements:
      w_quantized: the quantized version of w, represented as an uint8-
        or uint16-type numpy.ndarray.
      w_min: Minimum value of w, required for dequantization.
      w_max: Maximum value of w, required for dequantization.
  """
  if bits == 8:
    dtype = np.uint8
  elif bits == 16:
    dtype = np.uint16
  else:
    raise ValueError('Unsupported bits of quantization: %s' % bits)

  w_min = np.min(w)
  w_max = np.max(w)
  if w_max == w_min:
    raise ValueError('Cannot perform quantization because w has a range of 0')
  w_quantized = np.array(
       np.floor((w - w_min) / (w_max - w_min) * np.power(2, bits)), dtype)
  return w_quantized, w_min, w_max


def dequantize(w_quantized, w_min, w_max):
  """
  Simulate weight de-quantization.

  Args:
    w: (a numpy.ndarray) The weight to be quantized.
    bits: (int) number of bits used for the quantization: 8 or 16.

  Returns:
    A tuple with three elements:
      w_quantized: the quantized version of w, represented as an uint8-
        or uint16-type numpy.ndarray.
      w_min: Minimum value of w, required for dequantization.
      w_max: Maximum value of w, required for dequantization.
  """
  if w_quantized.dtype == np.uint8:
    bits = 8
  elif w_quantized.dtype == np.uint16:
    bits = 16
  else:
    raise ValueError(
        'Unsupported dtype in quantized values: %s' % w_quantized.dtype)
  return (w_min +
          w_quantized.astype(np.float64) / np.power(2, bits) * (w_max - w_min))


def main():
  # Number of points along the x-axis used to draw the sine wave.
  n_points = 1e6
  xs = np.linspace(-np.pi, np.pi, n_points).astype(np.float64)
  w = xs

  w_16bit = dequantize(*quantize(w, 16))
  w_8bit = dequantize(*quantize(w, 8))

  plot_delta = 1.2e-4
  plot_range = range(int(n_points * (0.5 - plot_delta)),
                     int(n_points * (0.5 + plot_delta)))

  plt.figure(figsize=(20, 6))
  plt.subplot(1, 3, 1)
  plt.plot(xs[plot_range], w[plot_range], '-')
  plt.title('Original (float32)', {'fontsize': 16})
  plt.xlabel('x')

  plt.subplot(1, 3, 2)
  plt.plot(xs[plot_range], w_16bit[plot_range], '-')
  plt.title('16-bit quantization', {'fontsize': 16})
  plt.xlabel('x')

  plt.subplot(1, 3, 3)
  plt.plot(xs[plot_range], w_8bit[plot_range], '-')
  plt.title('8-bit quantization', {'fontsize': 16})
  plt.xlabel('x')

  plt.show()


if __name__ == '__main__':
  main()
