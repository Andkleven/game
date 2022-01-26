import { NextPage } from "next";
import { css } from "@emotion/react";

import { Custom } from "../lib/Environments/examples/custom/index";
import { seed } from "../lib/utils/random";
import { MLPActorCritic } from "../lib/Models/ac";
import { PPO } from "../lib/Algos/ppo/index";
const env = new Custom();

// const main = async () => {
//   for (let episode = 0; episode < 100; episode++) {
//     let state = env.reset();
//     let steps = 0;
//     let rewards = 0;
//     while (true) {
//       const action = env.actionSpace.sample();
//       console.log(action);
//       const { reward, observation, done } = env.step(action);
//       rewards += reward;
//       state = observation;
//       if (done) break;
//       steps++;
//     }
//     console.log(`Step: ${steps}, Reward: ${rewards}`);
//   }
// };
const main = async () => {
  // seed for reproducibility
  seed(0);

  // create a makeEnv function. It should return a new environment object that the VPG algorithm will interact with
  const makeEnv = () => {
    return new Custom();
  };
  const env = makeEnv();

  // create the Actor Critic model
  const ac = new MLPActorCritic(
    env.observationSpace,
    env.actionSpace,
    [24, 48]
  );

  // create the vpg algorithm and define a actionToTensor function to discretize the actions produced by the Actor
  const ppo = new PPO(makeEnv, ac, {
    actionToTensor: (action) => {
      // Cartpole has a discrete action space whereas Actor Critic by default returns values according to the shape of the action space.
      // CartPole has an action space with shape [2] so we discretize the Actor by transforming its output into the argmax of it.
      return action.argMax(1);
    },
  });

  // define a evaluation function to be called at the end of every epoch
  const epochCallback = async ({ epoch }) => {
    let obs = env.reset();
    let rewards = 0;
    while (true) {
      const action = ppo.act(obs);
      const stepInfo = env.step(action);
      rewards += stepInfo.reward;
      if (epoch > 10) {
        // after 10 epochs, start rendering the evaluation onto a web viewer
        await env.render("web", { fps: 60, episode: epoch });
      }
      obs = stepInfo.observation;
      if (stepInfo.done) break;
    }
    console.log(`Episode ${epoch} - Eval Rewards: ${rewards}`);
  };

  // Uncomment the 2 lines below to train on 2 CPUs. Will train on forkCount + 1 cpus.
  // let forkCount = 1;
  // await RL.ct.fork(forkCount);

  // train the actor critic model with ppo
  ppo.train({
    // verbose: true,
    steps_per_epoch: 1000,
    epochs: 200,
    train_pi_iters: 80,
    train_v_iters: 80,
    epochCallback: epochCallback,
  });
};
// main();

const Home: NextPage = () => {
  return (
    <div
      css={css`
        background-color: red;
        color: red;
      `}
    >
      test
    </div>
  );
};

export default Home;
