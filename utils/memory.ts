import _ from "lodash";
import * as tf from "@tensorflow/tfjs";
import { tensorToArray } from "../v2/tools";

function zeroedObservation(observation) {
  if ("shape" in observation) {
    return tf.zeros(observation.shape);
  } else if ("__iter__" in observation) {
    let out: any[] = [];
    for (let x in observation) {
      out.push(zeroedObservation(x));
    }
    return out;
  } else {
    return 0;
  }
}
function getRandomInt(min: number, max: number) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
}
export function randint(min: number = 0, max: number, size: number = 1) {
  let array: number[] = [];
  for (let index = 0; index < size; index++) {
    array[index] = getRandomInt(min, max);
  }
  return array;
}
function randomIntFromInterval(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}
function sampleBatchIndexes(low: number, high: number, size: number): number[] {
  let batchIdxs: number[];
  console.log(low, high, size);
  if (high - low >= size) {
    const r: number[] = [];
    for (let i = low; i <= high; i++) {
      r.push(i);
    }
    batchIdxs = _.sampleSize(r, size);
  } else {
    console.warn(
      "Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!"
    );
    batchIdxs = Array.from({ length: size }, () =>
      randomIntFromInterval(low, high - 1)
    );
  }
  console.log(batchIdxs);
  if (batchIdxs.length !== size) {
    throw new Error("memory 57");
  }
  return batchIdxs;
}

export class Memory {
  windowLength: number;
  ignoreEpisodeBoundaries;
  recentObservations: number[][];
  recentTerminals: boolean[];

  constructor(windowLength: number, ignoreEpisodeBoundaries = false) {
    this.windowLength = windowLength;
    this.ignoreEpisodeBoundaries = ignoreEpisodeBoundaries;

    this.recentObservations = [];
    this.recentTerminals = [];
  }

  sample(
    batchSize: number,
    batchIdxs: number[] | null = null
  ): {
    state0: number[][];
    action: number;
    reward: number;
    state1: number[][];
    terminal1: any;
  }[] {
    throw "NotImplementedError";
  }
  push(
    observation: number[],
    action: number,
    reward: number,
    terminal: boolean,
    training = true
  ) {
    this.recentObservations.push(observation);
    this.recentTerminals.push(terminal);
  }
  getRecentState(currentObservation: number[]): [number[]] {
    let state: [number[]] = [currentObservation];
    let idx = this.recentObservations.length - 1;
    for (let offset = 0; this.windowLength - 1 < offset; offset++) {
      let currentIdx = idx - offset;
      let currentTerminal;
      if (currentIdx - 1 >= 0) {
        currentTerminal = this.recentTerminals[currentIdx - 1];
      } else {
        currentTerminal = false;
      }
      if (
        currentIdx < 0 ||
        (!this.ignoreEpisodeBoundaries && currentTerminal)
      ) {
        break;
      }
      state.unshift(this.recentObservations[currentIdx]);
    }
    while (state.length < this.windowLength) {
      state.unshift(zeroedObservation(state[0]));
    }
    return state;
  }
  getConfig() {
    let config = {
      windowLength: this.windowLength,
      ignoreEpisodeBoundaries: this.ignoreEpisodeBoundaries,
    };
    return config;
  }
}

export class SequentialMemory extends Memory {
  limit: number;
  actions: number[];
  rewards: number[];
  terminals: boolean[];
  observations: number[][];

  constructor(limit: number, windowLength: number) {
    super(windowLength);

    this.limit = limit;

    this.actions = [];
    this.rewards = [];
    this.terminals = [];
    this.observations = [];
  }
  sample(batchSize: number, batchIdxs: null | number[] = null) {
    if (this.nbEntries < this.windowLength + 2) {
      throw new Error("memory 57");
    }

    if (batchIdxs === null) {
      batchIdxs = sampleBatchIndexes(
        this.windowLength,
        this.nbEntries - 1,
        batchSize
      );
    }
    let newBatchIdxs = batchIdxs.map((batchId) => batchId + 1);
    if (
      tensorToArray(tf.tensor1d(newBatchIdxs).min())[0] <
      this.windowLength + 1
    ) {
      throw new Error("memory 153");
    }
    if (tensorToArray(tf.tensor1d(newBatchIdxs).max())[0] >= this.nbEntries) {
      throw new Error("memory 156");
    }
    if (batchIdxs.length !== batchSize) {
      throw new Error("memory 159");
    }
    let experiences: {
      state0: number[][];
      action: number;
      reward: number;
      state1: number[][];
      terminal1: any;
    }[] = [];
    for (let index = 0; index < batchIdxs.length; index++) {
      let idx = batchIdxs[index];
      let terminal0 = this.terminals[idx - 2];
      while (terminal0) {
        idx = sampleBatchIndexes(this.windowLength, this.nbEntries, 1)[0];
        terminal0 = this.terminals[idx - 2];
      }
      if (!(this.windowLength <= idx && idx <= this.nbEntries)) {
        throw new Error(
          `windowLength: ${this.windowLength}, idx: ${idx}, nbEntries: ${this.nbEntries}`
        );
      }
      let state0: number[][] = [this.observations[idx - 1]];
      for (let offset = 0; this.windowLength - 0 < offset; offset++) {
        let currentIdx = idx - 2 - offset;
        if (currentIdx < 1) {
          throw new Error("memory 175");
        }
        let currentTerminal = this.terminals[currentIdx - 1];
        if (currentTerminal && !this.ignoreEpisodeBoundaries) break;
        state0.unshift(this.observations[currentIdx]);
      }

      while (state0.length < this.windowLength) {
        state0.unshift(zeroedObservation(state0[0]));
      }
      let action = this.actions[idx - 1];
      let reward = this.rewards[idx - 1];
      let terminal1 = this.terminals[idx - 1];
      let state1: number[][] = [];
      state0.forEach((x, index) => {
        if (index === 0) {
          return;
        } else {
          state1.push(_.cloneDeep(x));
        }
      });

      state1.push(this.observations[idx]);
      if (state0.length !== this.windowLength) {
        throw new Error("memory 200");
      }
      if (state1.length !== state0.length) {
        throw new Error("memory 203");
      }
      experiences.push({
        state0,
        action,
        reward,
        state1,
        terminal1,
      });
    }
    if (experiences.length !== batchSize) {
      throw new Error("memory 209");
    }
    return experiences;
  }

  push(
    observation: number[],
    action: number,
    reward: number,
    terminal: boolean,
    training = true
  ) {
    super.push(observation, action, reward, terminal, training);

    if (training) {
      this.observations.push(observation);
      this.actions.push(action);
      this.rewards.push(reward);
      this.terminals.push(terminal);
    }
  }

  get nbEntries(): number {
    return this.observations.length;
  }

  getConfig() {
    let config = SequentialMemory.prototype.getConfig();
    config["limit"] = this.limit;
    return config;
  }
}
