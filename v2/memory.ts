import { sample } from "./tools";
import * as tf from "@tensorflow/tfjs";
import { shuffle } from "lodash";

type Experience = {
  state: number;
  action: number;
  nextState: number;
  reward: number;
};

export class ReplayBuffer {
  size: number;
  inputShape: number;
  counter = 0;
  stateBuffer: tf.TensorBuffer<tf.Rank.R2, "float32">;
  actionBuffer: tf.TensorBuffer<tf.Rank.R1, "int32">;
  rewardBuffer: tf.TensorBuffer<tf.Rank.R1, "float32">;
  newStateBuffer: tf.TensorBuffer<tf.Rank.R2, "float32">;
  terminalBuffer: tf.TensorBuffer<tf.Rank.R1, "bool">;
  constructor(size: number, inputShape: number) {
    this.size = size;
    this.inputShape = inputShape;
    this.stateBuffer = tf.buffer([this.size, inputShape], "float32");
    this.actionBuffer = tf.buffer([this.size], "int32");
    this.rewardBuffer = tf.buffer([this.size], "float32");
    this.newStateBuffer = tf.buffer([this.size, inputShape], "float32");
    this.terminalBuffer = tf.buffer([this.size], "bool");
  }
  async storeTuples(
    state: number[],
    action: number,
    reward: number,
    newState: number[],
    done: boolean
  ) {
    const idx = this.counter % this.size;
    for (let index = 0; index < this.inputShape; index++) {
      this.stateBuffer.set(state[index], idx, index);
    }
    this.actionBuffer.set(action, idx);
    this.rewardBuffer.set(reward, idx);
    for (let index = 0; index < this.inputShape; index++) {
      this.newStateBuffer.set(newState[index], idx, index);
    }
    this.terminalBuffer.set(done, idx);
    this.counter += 1;
  }
  sampleBuffer(batchSize: number) {
    const maxBuffer = Math.min(this.counter, this.size);
    const batch = shuffle(Array.from(Array(maxBuffer).keys())).slice(
      0,
      batchSize
    );

    const stateBatch = elementsByIndexes(
      this.stateBuffer.toTensor().arraySync(),
      batch
    );
    const actionBatch = elementsByIndexes(
      this.actionBuffer.toTensor().arraySync(),
      batch
    );
    const rewardBatch = elementsByIndexes(
      this.rewardBuffer.toTensor().arraySync(),
      batch
    );
    const newStateBatch = elementsByIndexes(
      this.newStateBuffer.toTensor().arraySync(),
      batch
    );
    const doneBatch = elementsByIndexes(
      this.terminalBuffer.toTensor().arraySync(),
      batch
    );

    return [
      tf.tensor(stateBatch),
      tf.tensor(actionBatch),
      tf.tensor(rewardBatch),
      tf.tensor(newStateBatch),
      tf.tensor(doneBatch),
    ];
  }
}

function elementsByIndexes<T>(array: T[], indexes: number[]) {
  const newArray: T[] = [];

  for (let i = 0; i < indexes.length; i++) {
    const index = indexes[i];
    newArray.push(array[index]);
  }
  return newArray;
}
