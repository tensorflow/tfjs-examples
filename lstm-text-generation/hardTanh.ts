import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, tidy} from '@tensorflow/tfjs-core';

export function hardTanh(x: Tensor): Tensor {
  return tidy(() => {
    const y = tfc.mul(.5, x);
    return tfc.clipByValue(y, -1, 1);
  });
}

export abstract class Activation extends serialization.Serializable {
  abstract apply(tensor: Tensor, axis?: number): Tensor;
  getConfig(): serialization.ConfigDict {
    return {};
  }
}

/**
 * Segment-wise linear approximation of tanh.
 */
export class HardTanh extends Activation {
  /** @nocollapse */
  static readonly className = 'hardTanh';
  apply(x: Tensor): Tensor {
    return hardTanh(x);
  }
}
serialization.registerClass(HardTanh);
