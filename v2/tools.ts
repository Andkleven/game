import * as tf from "@tensorflow/tfjs";

export function tensorToArray(v: tf.Tensor): number[] {
  return Array.from(v.dataSync());
}

export function matrix(m: number, n: number, value: number = 0) {
  return Array.from(
    {
      length: m,
    },
    () => new Array(n).fill(value)
  );
}

export function huberLoss(yTrue: tf.Tensor, yPred: tf.Tensor) {
  const x = tf.sub(yTrue, yPred);
  return tf.mul(0.5, tf.square(x));
}
function clippedMaskedError(inputs: number[][]) {
  const [yTrue, yPred, mask] = inputs;
  const x = tf.sub(yTrue, yPred);
  let loss = tf.mul(0.5, tf.square(x));
  loss = tf.mul(mask, loss);
  // return tf.sum(loss, -1);
  return loss;
}
export function getRandomInt(max: number) {
  return Math.floor(Math.random() * max);
}
export type ArgumentTypes<F extends Function> = F extends (
  ...args: infer A
) => any
  ? A
  : never;

export function sample<T>(sequence: T[], length: number) {
  const samples: T[] = [];
  for (let i = 0; i < length; i++) {
    const index = getRandomInt(sequence.length);
    samples.push(sequence[index]);
  }
  return samples;
}

export function isTensor(
  tensor:
    | tf.Tensor<tf.Rank>
    | tf.Tensor<tf.Rank>[]
    | tf.SymbolicTensor
    | tf.SymbolicTensor[]
) {
  if (tensor instanceof tf.SymbolicTensor || Array.isArray(tensor)) {
    throw new Error(`Tensor is ${typeof tensor}`);
  }
  return tensor;
}
