import * as tf from "@tensorflow/tfjs";
import { Agent } from "./agent";
import { BoltzmannQPolicy } from "../utils/policy";
import { SequentialMemory } from "../utils/memory";
import { DQNEnv } from "./env";

export async function run() {
  const env = new DQNEnv();

  // let action = env.sample();
  // for (let index = 0; index < 10; index++) {
  //   action = env.sample();
  //   const step = env.step(action);
  //   const observation = step[0];
  //   const r = step[1];
  //   const d = step[2];
  //   console.log(observation, r, d);
  // }

  // const model = tf.sequential();
  // model.add(tf.layers.flatten({ inputShape: [null, 2] }));
  // model.add(tf.layers.dense({ units: 16 }));
  // model.add(tf.layers.activation({ activation: "relu" }));
  // model.add(tf.layers.dense({ units: 16 }));
  // model.add(tf.layers.activation({ activation: "relu" }));
  // model.add(tf.layers.dense({ units: 16 }));
  // model.add(tf.layers.activation({ activation: "relu" }));
  // model.add(tf.layers.dense({ units: nbActions }));
  // model.add(tf.layers.activation({ activation: "linear" }));
  const dqn = new Agent({
    inputDim: env.getNumberOfObservation,
    numberActions: env.getNumberOfAction,
  });

  await dqn.trainModel(env, 100);
  console.log(12);

  // dqn.saveWeights("dqn_{}_weights.h5f");

  // dqn.test(env, { nbEpisodes: 5, nbMaxEpisodeSteps: 20 });
}
