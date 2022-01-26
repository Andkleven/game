import * as tf from "@tensorflow/tfjs";
import { Dense, Kwargs } from "@tensorflow/tfjs/dist/tf";
// import { model as clone } from "tfjs-clone";
import { EpsilonGreedyPolicy } from "./policy";
import { ReplayBuffer } from "./memory";
import { getRandomInt, isTensor } from "./tools";
import { sampleSize, mean } from "lodash";
import { Env } from "./env";

class DuelingDQN extends tf.Sequential {
  dense1: ReturnType<typeof tf.layers.dense>;
  dense2: ReturnType<typeof tf.layers.dense>;
  V: ReturnType<typeof tf.layers.dense>;
  A: ReturnType<typeof tf.layers.dense>;
  constructor(numberActions: number, fc1: number, fc2: number) {
    super();
    this.dense1 = tf.layers.dense({ units: fc1, activation: "relu" });
    this.dense2 = tf.layers.dense({ units: fc2, activation: "relu" });
    this.V = tf.layers.dense({ units: 1 });
    this.A = tf.layers.dense({ units: numberActions });
    super.add(tf.layers.flatten({ inputShape: [null, 2] }));
    super.add(this.dense1);
    super.add(this.dense2);
    super.add(this.V);
    super.add(this.A);
  }

  call(inputs: tf.Tensor | tf.Tensor[], kwargs: Kwargs) {
    let x = this.dense1.apply(inputs);
    x = this.dense2.apply(x);
    const V = this.V.apply(x);
    const A = this.A.apply(x);
    if (A instanceof tf.SymbolicTensor || Array.isArray(A)) {
      throw new Error(`A is ${typeof A}`);
    }
    if (V instanceof tf.SymbolicTensor || Array.isArray(V)) {
      throw new Error(` V is ${typeof V}`);
    }
    const avgA = tf.mean(A, 1, true);
    const Q = V.add(A.sub(avgA));

    return [Q, A];
  }
}
type IAgent = {
  model: tf.Sequential | null;
  numberActions: number;
  // memory: Memory;
  policy: EpsilonGreedyPolicy;
  gamma?: number;
  batchSize?: number;
  nbStepsWarmup?: number;
  trainInterval?: number;
  memoryInterval?: number;
  targetModelUpdate?: number;
};

// export class Agent {
//   model: tf.Sequential | null;
//   numberActions: number;
//   // memory: Memory;
//   policy: EpsilonGreedyPolicy;
//   gamma?: number;
//   batchSize?: number;
//   nbStepsWarmup?: number;
//   trainInterval?: number;
//   memoryInterval?: number;
//   targetModelUpdate?: number;
//   currentStep = 0;
//   constructor({
//     model,
//     numberActions,
//     policy,
//     gamma = 0.99,
//     batchSize = 32,
//     nbStepsWarmup = 1000,
//     trainInterval = 1,
//     memoryInterval = 1,
//     targetModelUpdate = 10000,
//   }: IAgent) {
//     this.model = model;
//     this.numberActions = numberActions;
//     this.policy = policy;
//     this.gamma = gamma;
//     this.batchSize = batchSize;
//     this.nbStepsWarmup = nbStepsWarmup;
//     this.trainInterval = trainInterval;
//     this.memoryInterval = memoryInterval;
//     this.targetModelUpdate = targetModelUpdate;
//   }
//   selectAction(state: number[], policyNet: tf.Sequential) {
//     const rate = this.policy.getExplorationRate(this.currentStep);
//     this.currentStep += 1;
//     if (rate > Math.random()) {
//       return getRandomInt(this.numberActions);
//     } else {
//       let predict = policyNet.predict(tf.tensor(state));
//       if (Array.isArray(predict)) {
//         return predict[0].argMax();
//       } else {
//         return predict.argMax();
//       }
//     }
//   }
// }

export class Agent {
  actionSpace: number[];
  discountFactor: number;
  epsilon: number;
  batchSize: number;
  epsilonDecay: number;
  epsilonFinal: number;
  updateRate: number;
  stepCounter = 0;
  buffer: ReplayBuffer;
  qModel: DuelingDQN | tf.LayersModel;
  qTargetModel: DuelingDQN | tf.LayersModel;
  constructor({
    inputDim,
    numberActions,
    learningRate = 0.00075,
    discountFactor = 0.99,
    epsilon = 1.0,
    batchSize = 64,
    epsilonDecay = 0.001,
    epsilonFinal = 0.01,
    updateRate = 120,
    buffer = 100000,
  }: {
    numberActions: number;
    inputDim: number;
    learningRate?: number;
    discountFactor?: number;
    epsilon?: number;
    batchSize?: number;
    epsilonDecay?: number;
    epsilonFinal?: number;
    updateRate?: number;
    buffer?: number;
  }) {
    this.actionSpace = Array.from(Array(10).keys());
    this.discountFactor = discountFactor;
    this.epsilon = epsilon;
    this.batchSize = batchSize;
    this.epsilonDecay = epsilonDecay;
    this.epsilonFinal = epsilonFinal;
    this.updateRate = updateRate;
    this.buffer = new ReplayBuffer(buffer, inputDim);
    this.qModel = new DuelingDQN(numberActions, 32, 32);
    this.qTargetModel = new DuelingDQN(numberActions, 32, 32);
    this.qModel.compile({
      optimizer: tf.train.adam(learningRate),
      loss: "meanSquaredError",
    });
    this.qTargetModel.compile({
      optimizer: tf.train.adam(learningRate),
      loss: "meanSquaredError",
    });
  }
  storeTuple(state, action, reward, newState, done) {
    this.buffer.storeTuples(state, action, reward, newState, done);
  }

  policy(observation: number[]) {
    let action: number;
    if (Math.random() < this.epsilon) {
      action = sampleSize(this.actionSpace, 1)[0];
    } else {
      const state = tf.tensor([observation]);
      const actions = this.qModel.apply(state);
      if (actions instanceof tf.SymbolicTensor || Array.isArray(actions)) {
        throw new Error(`actions is ${typeof actions}`);
      }
      action = tf.max(actions, 1).arraySync()[0];
    }

    return action;
  }
  async train() {
    if (this.buffer.counter < this.batchSize) {
      return;
    }
    if (this.stepCounter % this.updateRate === 0) {
      this.qTargetModel.setWeights(this.qModel.getWeights());
    }

    const [stateBatch, actionBatch, rewardBatch, newStateBatch, doneBatch] =
      this.buffer.sampleBuffer(this.batchSize);

    let qPredicted = this.qModel.apply(stateBatch);
    let qNext = this.qTargetModel.apply(newStateBatch);
    qNext = isTensor(qNext);
    qPredicted = isTensor(qPredicted);
    const qMaxNext = tf.max(qNext, 1, true);
    let qTarget = tf.clone(qPredicted);
    for (let idx = 0; idx < doneBatch.shape[0]; idx++) {
      let targetQValue = rewardBatch[idx];
      if (!doneBatch[idx]) {
        targetQValue += this.discountFactor * qMaxNext[idx];
      }
      const target = await qTarget.buffer();
      target.set(targetQValue, idx, actionBatch[idx]);
      qTarget = target.toTensor();
    }
    this.qModel.trainOnBatch(stateBatch, qTarget);
    if (this.epsilon > this.epsilonFinal) {
      this.epsilon = this.epsilon - this.epsilonDecay;
    } else {
      this.epsilon = this.epsilonFinal;
    }
    this.stepCounter += 1;
  }
  trainModel(env: Env, numberEpisodes: number) {
    const scores: number[] = [];
    const episodes: number[] = [];
    const avgScores: number[] = [];
    const obj: number[] = [];
    const goal = -110.0;
    let f = 0;
    for (let i = 0; i < numberEpisodes; i++) {
      let done = false;
      let score = 0.0;
      let state = env.reset();
      while (!done) {
        const action = this.policy(state);
        const [newState, reward, done] = env.step(action);
        score += reward;
        this.storeTuple(state, action, reward, newState, done);
        state = newState;
        this.train();
      }
      scores.push(score);
      obj.push(goal);
      episodes.push(i);
      const avgScore = mean(scores.slice(-100));
      avgScores.push(avgScore);
      console.log(
        `Episode ${i}/${numberEpisodes}, Score: ${score} (${this.epsilon}), AVG Score: ${avgScore}`
      );
      this.qModel.save;
      if (avgScore >= -110.0 && score >= -108) {
        // await this.qModel.save(`saved_networks/duelingdqn_model${f}`)
        // this.qModel.saveWeights(
        //     `saved_networks/duelingdqn_model${f}/net_weights${f}.h5`)
        f += 1;
        console.log("Network saved");
      }
    }
  }
  async test(env: Env, numberEpisodes, fileType, file) {
    if (fileType == "tf") {
      this.qModel = await tf.loadLayersModel(file);
    } else if (fileType == "h5") {
      this.trainModel(env, 5);
      this.qModel.loadWeights(file);
    }
    this.epsilon = 0.0;
    const scores: number[] = [];
    const episodes: number[] = [];
    const avgScores: number[] = [];
    const obj: number[] = [];
    let goal = -110.0;
    let score = 0.0;
    for (let i = 0; i < numberEpisodes; i++) {
      let state = env.reset();
      let done = false;
      let episodeScore = 0.0;
      while (!done) {
        let action = this.policy(state);
        const [newState, reward] = env.step(action);
        episodeScore += reward;
        state = newState;
      }
      score += episodeScore;
      scores.push(episodeScore);
      obj.push(goal);
      episodes.push(i);
      const avgScore = mean(scores.slice(-100));
      avgScores.push(avgScore);
    }
  }
}
