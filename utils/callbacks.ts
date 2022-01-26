import * as tf from "@tensorflow/tfjs";
import { Env } from "../v2/env";

export class Callback extends tf.Callback {
  constructor() {
    super();
  }
  onEpisodeBegin(episode, logs = {}) {}

  onEpisodeEnd(episode, logs = {}) {}

  onStepBegin(step, logs = {}) {}

  onActionBegin(action, logs = {}) {}

  onActionEnd(action, logs = {}) {}
  onStepEnd(
    step,
    logs: {
      action: number;
      observation: number[];
      reward: number;
      metrics?: number[];
      episode: number;
    }
  ) {}
}

export class TrainEpisodeLogger extends Callback {
  trainStart: number = 0;
  episodeStart: { number?: number } = {};
  observations: { number?: number[][] } = {};
  rewards: { number?: number[][] } = {};
  actions: { number?: number[][] } = {};
  metrics: { number?: number[][] } = {};
  step: number = 0;
  params: { [key: string]: number | string } = {};
  constructor() {
    super();
  }

  async onTrainBegin() {
    this.trainStart = Date.now();
    console.log(`Training for ${this.params["nbSteps"]} steps ...`);
  }

  async onTrainEnd() {
    const duration = Date.now() - this.trainStart;
    console.log(`done, took ${duration} seconds`);
  }

  onEpisodeBegin(episode: number) {
    this.episodeStart[episode] = Date.now();
    this.observations[episode] = [];
    this.rewards[episode] = [];
    this.actions[episode] = [];
    this.metrics[episode] = [];
  }

  onEpisodeEnd(episode: number) {
    const duration = Date.now() - this.episodeStart[episode];
    let episodeSteps = this.observations[episode].length;

    const template = `{step: ${this.step}: episode: ${
      episode + 1
    }, duration: ${duration}s, episode steps: ${episodeSteps}, steps per second: ${
      episodeSteps / duration
    }, episode reward: ${this.rewards[episode].reduce(
      (a: number, b: number) => a + b
    )}`;

    delete this.episodeStart[episode];
    delete this.observations[episode];
    delete this.rewards[episode];
    delete this.actions[episode];
    delete this.metrics[episode];
  }

  onStepEnd(
    step,
    logs: {
      action: number;
      observation: number[];
      reward: number;
      metrics?: number[];
      episode: number;
    }
  ) {
    const episode = logs["episode"];
    this.observations[episode].push(logs["observation"]);
    this.rewards[episode].push(logs["reward"]);
    this.actions[episode].push(logs["action"]);
    this.metrics[episode].push(logs["metrics"]);
    this.step += 1;
  }
}

export class CallbackList extends tf.CallbackList {
  callbacks: (Callback | tf.History)[];
  constructor(callbacks: (Callback | tf.History)[], queueLength?: number) {
    super(callbacks, queueLength);
    this.callbacks = callbacks;
  }
  // _setEnv(env: Env) {
  //   for (let index = 0; index < this.callbacks.length; index++) {
  //     const callback = this.callbacks[index];
  //     if (callback instanceof tf.Callback) {
  //       callback._setEnv(env);
  //     }
  //   }
  // }
  onEpisodeBegin(episode: number, logs = {}) {
    for (let index = 0; index < this.callbacks.length; index++) {
      const callback = this.callbacks[index];
      if ("onEpisodeBegin" in callback) {
        callback.onEpisodeBegin(episode, logs);
      } else {
        callback.onEpochBegin(episode, logs);
      }
    }
  }

  onEpisodeEnd(episode: number, logs = {}) {
    for (let index = 0; index < this.callbacks.length; index++) {
      const callback = this.callbacks[index];
      if ("onEpisodeEnd" in callback) {
        callback.onEpisodeEnd(episode, logs);
      } else {
        callback.onEpochEnd(episode, logs);
      }
    }
  }
  onStepBegin(step: number, logs = {}) {
    for (let index = 0; index < this.callbacks.length; index++) {
      const callback = this.callbacks[index];
      if ("onStepBegin" in callback) {
        callback.onStepBegin(step, logs);
      } else {
        callback.onBatchBegin(step, logs);
      }
    }
  }
  onStepEnd(
    step: number,
    logs: {
      action: number;
      observation: number[];
      reward: number;
      metrics?: number[];
      episode: number;
    }
  ) {
    for (let index = 0; index < this.callbacks.length; index++) {
      const callback = this.callbacks[index];
      if ("onStepEnd" in callback) {
        callback.onStepEnd(step, logs);
      } else {
        callback.onBatchEnd(step, logs);
      }
    }
  }

  onActionBegin(action: number, logs = {}) {
    for (let index = 0; index < this.callbacks.length; index++) {
      const callback = this.callbacks[index];
      if ("onActionBegin" in callback) {
        callback.onActionBegin(action, logs);
      }
    }
  }

  onActionEnd(action: number, logs = {}) {
    for (let index = 0; index < this.callbacks.length; index++) {
      const callback = this.callbacks[index];
      if ("onActionEnd" in callback) {
        callback.onActionEnd(action, logs);
      }
    }
  }
}

export class TestLogger extends Callback {
  constructor() {
    super();
  }

  async onTrainBegin(logs = {}) {
    console.log(`Testing for ${this.params["nbEpisodes"]} episodes ...`);
  }
  onEpisodeEnd(episode: number, logs = {}) {
    console.log(
      `Episode ${episode + 1}: reward: ${logs["episodeReward"]}, steps: ${
        logs["nbSteps"]
      }`
    );
  }
}
