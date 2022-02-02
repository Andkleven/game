/**
 * Various tooling for computing values across Node.js processes to take advantage of multi-core systems via the cluster library
 *
 * Primary / Master process aggregates messages received from other workers
 */
import * as tf from "@tensorflow/tfjs";
tf.ENV.set("WEBGL_PACK", false);
import { OPS } from "./ops";
import * as np from "../np";

/**
 * Compute various statistics of a registered variable
 */
// TODO: This can be optimized significantly. There is a lot of waiting here...
export const statisticsScalar = async (
  x: tf.Tensor,
  exclude: {
    mean?: boolean;
    std?: boolean;
    max?: boolean;
    min?: boolean;
  } = {},
  asTensor = false
) => {
  let maxv;
  if (!exclude.max) {
    maxv = x.max();
  }
  let minv;
  if (!exclude.max) {
    minv = x.min();
  }
  let global_sum;
  let global_n;
  let mean;
  let std;
  if (!exclude.mean || !exclude.std) {
    global_sum = x.sum();
    global_n = x.shape[0];
    mean = global_sum.div(global_n);

    if (!exclude.std) {
      const global_sum_sq = x.sub(mean).pow(2).sum();
      std = global_sum_sq.div(global_n).sqrt();
    }
  }
  const data = {
    max: maxv,
    min: minv,
    mean,
    std,
  };

  if (asTensor) {
    return data;
  }
  if (data.mean) {
    data.mean = np.tensorLikeToJSArray(data.mean);
  }
  if (data.std) {
    data.std = np.tensorLikeToJSArray(data.std);
  }
  if (data.min) {
    data.min = np.tensorLikeToJSArray(data.min);
  }
  if (data.max) {
    data.max = np.tensorLikeToJSArray(data.max);
  }
  return data;
};

/**
 *
 * @param x - tensor with shape [...shape]
 * @param rest - tensor with shape [P, ...shape]
 */
const handleOp = async (x: tf.Tensor, rest: tf.Tensor, op: OPS) => {
  switch (op) {
    case OPS.SUM:
      return x.add(rest.sum(0));
    case OPS.GATHER:
      return x.concat(rest);
    case OPS.MAX:
      return x.max().concat(rest.max()).max();
    case OPS.MIN:
      return x.min().concat(rest.min()).min();
  }
};

/**
 *
 * @param x - tensor with shape [...shape]
 * @param rest - tensor with shape [P, ...shape]
 */
const handleOpNumber = async (x: number, rest: number[], op: OPS) => {
  switch (op) {
    case OPS.SUM: {
      let val = 0;
      for (const v of rest) {
        val += v;
      }
      return x + val;
    }
    default:
      throw new Error(`${op} is invalid op`);
  }
};
