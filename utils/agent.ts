import * as tf from "@tensorflow/tfjs";
import {
  TrainEpisodeLogger,
  CallbackList,
  TestLogger,
  Callback,
} from "./callbacks";
import _ from "lodash";
import {
  AdditionalUpdatesOptimizer,
  ArgumentTypes,
  antirectifier,
  tensorToArray,
  matrix,
} from "../v2/tools";
import { Env } from "../v2/env";
import { randint, Memory } from "./memory";
import { EpsGreedyQPolicy, GreedyQPolicy, Policy } from "./policy";

type ConfigTest = {
  nbEpisodes?: number;
  actionRepetition?: number;
  nbMaxEpisodeSteps?: number | null;
};
const meanQ = (yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor => {
  return tf.mean(tf.max(yPred, -1));
};
class Agent {
  training: boolean;
  step: number;
  compiled = false;
  model: tf.Sequential | null = null;
  constructor() {
    this.training = false;
    this.step = 0;
  }

  getConfig() {
    return {};
  }

  async fit(
    env: Env,
    nbSteps: number,
    actionRepetition = 1,
    nbMaxEpisodeSteps = null
  ) {
    if (!this.compiled) {
      console.warn(
        "Your tried to fit your agent but it hasn't been compiled yet. Please call `compile()` before `fit()`."
      );
    }
    if (actionRepetition < 1) {
      console.warn(`actionRepetition must be >= 1, is ${actionRepetition}`);
    }
    this.training = true;

    let callbacks: (Callback | tf.History)[] = [new TrainEpisodeLogger()];

    const history = new tf.History();
    callbacks = callbacks.concat([history]);
    const callbackList = new CallbackList(callbacks);
    if (this.model === null) {
      throw new Error("agent, 58");
    }
    callbackList.setModel(this.model);

    let params = {
      nbSteps: nbSteps,
    };

    callbackList.setParams(params);

    callbackList.onTrainBegin();
    // int16
    let episode = 0;
    // int16
    this.step = 0;
    let observation: number[] | null = null;
    let episodeReward: number | null = null;
    let episodeStep: number | null = null;
    let didAbort = 0;
    let action: number;
    let done: boolean;
    let reward: number;

    try {
      while (this.step < nbSteps) {
        if (observation === null) {
          callbackList.onEpisodeBegin(episode);
          // int16
          episodeStep = 0;
          // float32
          episodeReward = 0;
          this.resetStates();
          observation = _.cloneDeep(env.reset());
          if (observation === null) {
            throw new Error("agent 100");
          }
        }
        if (
          episodeReward === null ||
          episodeStep === null ||
          observation === null
        ) {
          throw new Error("Error");
        }

        callbackList.onStepBegin(episodeStep);
        action = this.forward(observation);
        // float32
        reward = 0;
        done = false;
        let r: number;
        for (let index = 0; index < actionRepetition; index++) {
          callbackList.onActionBegin(action);
          const step = env.step(action);
          observation = step[0];
          r = step[1];
          done = step[2];
          observation = _.cloneDeep(observation);
          callbackList.onActionEnd(action);
          reward += r;
          if (done) {
            break;
          }
        }

        if (
          nbMaxEpisodeSteps &&
          episodeStep !== null &&
          episodeStep >= nbMaxEpisodeSteps - 1
        ) {
          done = true;
        }
        const metrics = await this.backward(reward, done);
        if (episodeReward === null) {
          episodeReward = 0;
        }
        episodeReward += reward;
        const stepLogs = {
          action,
          observation,
          reward,
          metrics,
          episode,
        };
        callbackList.onStepEnd(episodeStep, stepLogs);

        if (episodeStep !== null) {
          episodeStep += 1;
        }
        this.step += 1;

        if (done) {
          if (observation !== null) {
            this.forward(observation);
          }
          this.backward(0, false);

          const episodeLogs = {
            episodeReward: episodeReward,
            nbSteps: episodeStep,
            observation: observation,
          };
          callbackList.onEpisodeEnd(episode, episodeLogs);
          episode += 1;
          observation = null;
          episodeStep = null;
          episodeReward = null;
        }
      }
    } catch (error) {
      console.error(error);
      didAbort = 1;
    }
    callbackList.onTrainEnd({ didAbort });
    return history;
  }

  test(env: Env, config: ConfigTest) {
    const {
      nbEpisodes = 1,
      actionRepetition = 1,
      nbMaxEpisodeSteps = null,
    } = config;
    if (!this.compiled) {
      console.warn(
        "Your tried to test your agent but it hasn't been compiled yet. Please call `compile()` before `test()`."
      );
    }
    if (actionRepetition < 1) {
      console.warn(`actionRepetition must be >= 1, is ${actionRepetition}`);
    }

    this.training = false;
    this.step = 0;

    let callbacks: (Callback | tf.History)[] = [new TestLogger()];

    callbacks = callbacks.concat([new TestLogger()]);

    const history = new tf.History();
    callbacks = callbacks.concat([history]);
    const callbackList = new CallbackList(callbacks);

    if (this.model === null) {
      throw new Error("agent, 186");
    }
    callbackList.setModel(this.model);

    const params = {
      nbEpisodes: nbEpisodes,
    };
    callbackList.setParams(params);

    callbackList.onTrainBegin();
    for (let episode = 0; episode < nbEpisodes; episode++) {
      callbackList.onEpisodeBegin(episode);
      let episodeReward = 0;
      let episodeStep = 0;

      this.resetStates();
      let observation = _.cloneDeep(env.reset());
      if (observation === null) {
        throw new Error("agent 218");
      }
      let r: number;
      let action: number;
      let reward: number;
      let done = false;
      while (!done) {
        callbackList.onStepBegin(episodeStep);

        action = this.forward(observation);
        reward = 0;
        let d: boolean;
        for (let index = 0; index < actionRepetition; index++) {
          callbackList.onActionBegin(action);
          const step = env.step(action);
          observation = step[0];
          r = step[1];
          d = step[2];
          observation = _.cloneDeep(observation);

          callbackList.onActionEnd(action);
          reward += r;

          if (d) {
            done = true;
            break;
          }
        }

        if (
          nbMaxEpisodeSteps &&
          episodeStep &&
          episodeStep >= nbMaxEpisodeSteps - 1
        ) {
          done = true;
        }
        this.backward(reward, done);
        episodeReward += reward;
        const stepLogs = {
          action,
          observation,
          reward,
          episode,
        };
        callbackList.onStepEnd(episodeStep, stepLogs);
        episodeStep += 1;
        this.step += 1;
      }

      this.forward(observation);
      this.backward(0, false);
      const episodeLogs = {
        episodeReward: episodeReward,
        nbSteps: episodeStep,
        observation: observation,
      };
      callbackList.onEpisodeEnd(episode, episodeLogs);
    }

    callbackList.onTrainEnd();

    return history;
  }
  resetStates() {}

  forward(observation: number[]): number {
    throw new Error("NotImplementedError");
  }

  backward(reward: number, terminal: boolean): Promise<number[]> {
    throw new Error("NotImplementedError");
  }

  compile(
    optimizer: tf.Optimizer,
    model: tf.Sequential,
    inputMetrics: string[]
  ) {
    throw new Error("NotImplementedError");
  }

  loadWeights(filepath) {
    throw new Error("NotImplementedError");
  }

  saveWeights(filepath: number, overwrite = false) {
    throw new Error("NotImplementedError");
  }

  get layers(): tf.layers.Layer[] {
    throw new Error("NotImplementedError");
  }

  get metricsNames(): string[] {
    return [];
  }
}

type IAbstractDQNAgent = {
  nbActions: number;
  memory: Memory;
  gamma?: number;
  batchSize?: number;
  nbStepsWarmup?: number;
  trainInterval?: number;
  memoryInterval?: number;
  targetModelUpdate?: number;
};

class AbstractDQNAgent extends Agent {
  nbActions: number;
  memory: Memory;
  gamma: number;
  batchSize: number;
  nbStepsWarmup: number;
  trainInterval: number;
  memoryInterval: number;
  targetModelUpdate: number;

  constructor({
    nbActions,
    memory,
    gamma = 0.99,
    batchSize = 32,
    nbStepsWarmup = 1000,
    trainInterval = 1,
    memoryInterval = 1,
    targetModelUpdate = 10000,
  }: IAbstractDQNAgent) {
    super();

    if (targetModelUpdate < 0) {
      throw new Error("`targetModelUpdate` must be >= 0.");
    } else if (targetModelUpdate >= 1) {
      targetModelUpdate = Math.floor(targetModelUpdate);
    }

    this.nbActions = nbActions;
    this.gamma = gamma;
    this.batchSize = batchSize;
    this.nbStepsWarmup = nbStepsWarmup;
    this.trainInterval = trainInterval;
    this.memoryInterval = memoryInterval;
    this.targetModelUpdate = targetModelUpdate;
    this.memory = memory;

    this.compiled = false;
  }

  computeBatchQValues(stateBatch: [[number[]]]) {
    const qValues = this.model?.predict(tf.tensor(stateBatch));
    if (Array.isArray(qValues) || qValues === undefined) {
      throw new Error("agent 364");
    }
    if (
      JSON.stringify(qValues?.shape) !==
      JSON.stringify([stateBatch.length, this.nbActions])
    ) {
      throw new Error("agent 376");
    }
    return Array.from(qValues?.dataSync());
  }

  computeQValues(state: [number[]]) {
    const qValues = this.computeBatchQValues([state]);
    return qValues;
  }

  getConfig() {
    return {
      nbActions: this.nbActions,
      gamma: this.gamma,
      batchSize: this.batchSize,
      nbStepsWarmup: this.nbStepsWarmup,
      trainInterval: this.trainInterval,
      memoryInterval: this.memoryInterval,
      targetModelUpdate: this.targetModelUpdate,
      memory: getObjectConfig(this.memory),
    };
  }
}

function getObjectConfig(o: Memory | tf.Sequential | null | Policy) {
  if (o === null) {
    return null;
  }

  const config = {
    className: o.constructor.name,
    config: o.getConfig(),
  };
  return config;
}

export class DQNAgent extends AbstractDQNAgent {
  targetModel: tf.Sequential | null = null;
  trainableModel: tf.LayersModel | null = null;
  model: tf.Sequential;
  recentObservation;
  recentAction;
  __policy!: Policy;
  __testPolicy!: Policy;
  constructor({
    model,
    policy,
    testPolicy = null,
    ...props
  }: {
    model: tf.Sequential;
    policy: Policy;
    testPolicy?: null | Policy;
  } & IAbstractDQNAgent) {
    super(props);
    if (
      JSON.stringify(model.outputs[0].shape) !==
      JSON.stringify([null, this.nbActions])
    ) {
      throw new Error(
        `Model output "${model.output}" has invalid shape. DQN expects a model that has one dimension for each action, in this case ${this.nbActions}.`
      );
    }

    this.model = model;
    if (policy === null) {
      policy = new EpsGreedyQPolicy();
    }
    if (testPolicy === null) {
      testPolicy = new GreedyQPolicy();
    }
    this.policy = policy;
    this.testPolicy = testPolicy;

    this.resetStates();
  }
  getConfig() {
    const config = AbstractDQNAgent.prototype.getConfig();
    config["model"] = getObjectConfig(this.model);
    config["policy"] = getObjectConfig(this.policy);
    config["testPolicy"] = getObjectConfig(this.testPolicy);
    if (this.compiled) {
      config["targetModel"] = getObjectConfig(this.targetModel);
    }
    return config;
  }

  async compile(
    optimizer: tf.Optimizer,
    model: tf.Sequential,
    inputMetrics: string[]
  ) {
    const metrics: Exclude<
      ArgumentTypes<tf.LayersModel["compile"]>[0]["metrics"],
      undefined
    > = [...inputMetrics, meanQ];

    this.targetModel = model;
    this.targetModel.compile({
      optimizer: "sgd",
      loss: tf.losses.meanSquaredError,
    });
    this.model.compile({ optimizer: "sgd", loss: tf.losses.meanSquaredError });

    if (this.targetModelUpdate < 1) {
      const updates = getSoftTargetModelUpdates(
        this.targetModel,
        this.model,
        this.targetModelUpdate
      );
      optimizer = new AdditionalUpdatesOptimizer(optimizer, updates);
    }

    const yPred = this.model.output;
    if (Array.isArray(yPred)) {
      throw new Error("Is array");
    }
    const yTrue = tf.layers.input({
      name: "yTrue",
      shape: [this.nbActions],
    });
    const mask = tf.layers.input({
      name: "mask",
      shape: [this.nbActions],
    });
    // const lossOut = clippedMaskedError(yTrue, yPred, mask);
    let ins: tf.SymbolicTensor[];
    if (Array.isArray(this.model.input)) {
      ins = this.model.input;
    } else {
      ins = [this.model.input];
    }
    const lossOut = antirectifier().apply([yTrue, yPred, mask]);
    if (Array.isArray(lossOut) || lossOut instanceof tf.Tensor) {
      throw new Error("agent 516");
    }
    const inputs = [...ins, yTrue, mask];
    const outputs = [lossOut, yPred];
    const trainableModel = tf.model({ inputs: inputs, outputs: outputs });
    if (trainableModel.outputNames.length !== 2) {
      throw new Error("agent 395");
    }
    const combinedMetrics = { [trainableModel.outputNames[1]]: metrics };
    const losses = [
      (yTrue: tf.Tensor, yPred: tf.Tensor) => yPred,
      (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.zerosLike(yPred),
    ];

    trainableModel.compile({
      optimizer: optimizer,
      loss: losses,
      metrics: combinedMetrics,
    });
    this.trainableModel = trainableModel;

    this.compiled = true;
  }
  loadWeights(filepath) {
    this.model.loadWeights(filepath);
    this.updateTargetModelHard();
  }

  saveWeights(filepath) {
    console.warn(filepath);
    // this.model.save(filepath);
  }
  resetStates() {
    this.recentAction = null;
    this.recentObservation = null;
    if (this.compiled) {
      if (this.targetModel === null) {
        throw new Error("targetModel is null");
      }
      this.model.resetStates();
      this.targetModel.resetStates();
    }
  }

  updateTargetModelHard() {
    if (this.targetModel === null) {
      throw new Error("targetModel is null");
    }
    this.targetModel.setWeights(this.model.getWeights());
  }
  forward(observation: number[]) {
    const state = this.memory.getRecentState(observation);
    const qValues = this.computeQValues(state);
    let action: number;
    if (this.training) {
      action = this.policy.selectAction(qValues);
    } else {
      action = this.testPolicy.selectAction(qValues);
    }

    this.recentObservation = observation;
    this.recentAction = action;

    return action;
  }

  async backward(reward: number, terminal: boolean) {
    if (this.step % this.memoryInterval === 0) {
      this.memory.push(
        this.recentObservation,
        this.recentAction,
        reward,
        terminal,
        this.training
      );
    }

    const metrics = new Array(this.metricsNames.length).map(() => NaN);
    if (!this.training) {
      return metrics;
    }

    if (this.step > this.nbStepsWarmup && this.step % this.trainInterval == 0) {
      const experiences = this.memory.sample(this.batchSize);
      if (experiences.length !== this.batchSize) {
        throw new Error("agent 395");
      }

      let state0Batch: number[][] = [];
      let rewardBatch: number[] = [];
      let actionBatch: number[] = [];
      let terminal1Batch: any = [];
      let state1Batch: number[][] = [];
      for (let index = 0; index < experiences.length; index++) {
        const element: any = experiences[index];
        state0Batch.push(element.state0);
        state1Batch.push(element.state1);
        rewardBatch.push(element.reward);
        actionBatch.push(element.action);
        if (element.terminal1) {
          terminal1Batch.push(0);
        } else {
          terminal1Batch.push(1);
        }
      }

      const terminal1BatchTensor = tf.tensor(terminal1Batch);
      const rewardBatchTensor = tf.tensor(rewardBatch);
      if (
        JSON.stringify(rewardBatchTensor.shape) !==
        JSON.stringify([this.batchSize])
      ) {
        throw new Error("agent 624");
      }
      if (
        JSON.stringify(terminal1BatchTensor.shape) !==
        JSON.stringify(rewardBatchTensor.shape)
      ) {
        throw new Error(
          `terminal1BatchTensor.shape !== rewardBatchTensor.shape, ${rewardBatchTensor.shape}, ${terminal1BatchTensor.shape}`
        );
      }
      if (this.targetModel === null) {
        throw new Error("agent 633");
      }
      const targetQValues = this.targetModel.predictOnBatch(
        tf.tensor(state1Batch)
      );
      if (
        Array.isArray(targetQValues) ||
        JSON.stringify(targetQValues.shape) !==
          JSON.stringify([this.batchSize, this.nbActions])
      ) {
        throw new Error("agent 640");
      }
      const qBatch = tf.max(targetQValues, 1).flatten();
      if (
        Array.isArray(targetQValues) ||
        JSON.stringify(qBatch.shape) !== JSON.stringify([this.batchSize])
      ) {
        throw new Error("agent 644");
      }

      let targets = matrix(this.batchSize, this.nbActions);
      let dummyTargets: number[][] = Array.from({ length: this.batchSize }).map(
        () => [0]
      );
      let masks = matrix(this.batchSize, this.nbActions);

      let discountedRewardBatch = tf.mul(this.gamma, qBatch);
      discountedRewardBatch = tf.mul(
        discountedRewardBatch,
        terminal1BatchTensor
      );
      if (
        JSON.stringify(discountedRewardBatch.shape) !==
        JSON.stringify(rewardBatchTensor.shape)
      ) {
        throw new Error("agent 654");
      }
      const Rs = tensorToArray(
        tf.add(rewardBatchTensor, discountedRewardBatch)
      );
      for (let index = 0; index < actionBatch.length; index++) {
        const R = Rs[index];
        const action = actionBatch[index];
        targets[index][action] = R;
        dummyTargets[index] = [R];
        masks[index][action] = 1;
      }

      let ins: number[][] | number[][][];
      if (Array.isArray(this.model.output)) {
        ins = state0Batch;
      } else {
        ins = [state0Batch];
      }
      if (this.trainableModel === null) {
        throw new Error("agent 673");
      }
      // const t = tf.tensor([...ins, targets, masks]);
      const t = [tf.tensor(ins[0]), tf.tensor(targets), tf.tensor(masks)];
      let metrics = await this.trainableModel.trainOnBatch(t, [
        tf.tensor(dummyTargets),
        tf.tensor(targets),
      ]);
      if (!Array.isArray(metrics)) {
        throw new Error("agent 673");
      }
      for (let index = 0; index < (metrics?.length ?? metrics); index++) {
        const metric = metrics[index];
        if (!(index in [1, 2])) {
          metrics.push(metric);
        }
      }
    }
    if (
      this.targetModelUpdate >= 1 &&
      this.step % this.targetModelUpdate == 0
    ) {
      this.updateTargetModelHard();
    }

    return metrics;
  }

  get layers() {
    return _.cloneDeep(this.model.layers);
  }

  get metricsNames() {
    if (this.trainableModel?.outputNames?.length !== 2) {
      throw new Error("agent 703");
    }
    const dummyOutputName = this.trainableModel?.outputNames[1];
    let modelMetrics: string[] = [];
    for (
      let index = 0;
      index < (this.trainableModel?.metricsNames?.length ?? 0);
      index++
    ) {
      const name = this.trainableModel?.metricsNames[index];
      if (!(index in [1, 2]) && name !== undefined) {
        modelMetrics.push(name);
      }
    }
    modelMetrics = modelMetrics.map((name) =>
      name.replace(dummyOutputName + "_", "")
    );

    let names = [...modelMetrics, ..._.cloneDeep(this.policy.metricsNames)];
    return names;
  }
  get policy() {
    return this.__policy;
  }

  set policy(policy) {
    this.__policy = policy;
    this.__policy.setAgent(this);
  }

  get testPolicy() {
    return this.__testPolicy;
  }

  set testPolicy(policy) {
    this.__testPolicy = policy;
    this.__testPolicy.setAgent(this);
  }
}

function getSoftTargetModelUpdates(
  target: tf.Sequential,
  source: tf.Sequential,
  tau: number
) {
  let updates: (number | tf.Tensor)[][] = [];
  for (let index = 0; index < target.getWeights().length; index++) {
    const tw = target.getWeights()[index];
    const sw = source.getWeights()[index];
    updates.push([tw, tf.mul(tf.mul(tf.add(sw, 1 - tau), tw), tau)]);
  }
  return updates;
}
