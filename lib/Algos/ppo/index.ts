import { Agent } from "../../Agent";
import { Environment } from "../../Environments";
import { Space } from "../../Spaces/box";
import * as random from "../../utils/random";
import * as tf from "@tensorflow/tfjs";
tf.ENV.set("WEBGL_PACK", false);
import { PPOBuffer, PPOBufferComputations } from "../../Algos/ppo/buffer";
import { DeepPartial } from "../../utils/types";
import { deepMerge } from "../../utils/deep";
import * as np from "../../utils/np";
import * as ct from "../../utils/clusterTools";
import { ActorCritic } from "../../Models/ac";
import { Action, EpochCallback, StepCallback } from "../../Agent/type";
import pino from "pino";
import { NdArray } from "numjs";

const log = pino({
  prettyPrint: {
    colorize: true,
  },
});
export interface PPOConfigs<Observation, Action> {
  /** Converts observations to batchable tensors of shape [1, ...observation shape] */
  obsToTensor: (state: Observation) => tf.Tensor;
  /** Converts actor critic output tensor to tensor that works with environment. Necessary if in discrete action space! */
  actionToTensor: (action: tf.Tensor) => TensorLike;
  /** Optional act function to replace the default act */
  act?: (obs: Observation) => Action;
}
export interface PPOTrainConfigs {
  stepCallback: StepCallback;
  epochCallback: EpochCallback;
  piLr: number;
  vfLr: number;
  /** How frequently in terms of total steps to save the model. This is not used if saveDirectory is not provided */
  ckptFreq: number;
  /** path to store saved models in. */
  savePath?: string;
  saveLocation?: TFSaveLocations;
  epochs: number;
  verbosity: string;
  gamma: number;
  lam: number;
  targetKl: number;
  clipRatio: number;
  trainVIters: number;
  trainPiIters: number;
  stepsPerEpoch: number;
  /** maximum length of each trajectory collected */
  maxEpLen: number;
  seed: number;
  name: string;
}

export class PPO<
  Observation extends NdArray<number>,
  ObservationSpace extends Space<Observation>,
  ActionSpace extends Space<Action>
> extends Agent<Observation, Action> {
  public configs: PPOConfigs<Observation, Action> = {
    obsToTensor: (obs: Observation) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if observation space is dict. if observation space is dict, user needs to override this.
      const tensor = np.tensorLikeToTensor(obs);
      return tensor.reshape([1, ...tensor.shape]);
    },
    actionToTensor: (action: tf.Tensor) => {
      // eslint-disable-next-line
      // @ts-ignore - let this throw an error, which can happen if action space is dict. if action space is dict, user needs to override this.
      return action;
    },
  };

  public env: Environment<
    ObservationSpace,
    ActionSpace,
    Observation,
    any,
    Action,
    number
  >;

  /** Converts observations to batchable tensors of shape [1, ...observation shape] */
  private obsToTensor: (obs: Observation) => tf.Tensor;
  private actionToTensor: (action: tf.Tensor) => TensorLike;
  private stopValue = false;
  private done = true;

  constructor(
    /** function that creates environment for interaction */
    public makeEnv: () => Environment<
      ObservationSpace,
      ActionSpace,
      Observation,
      any,
      Action,
      number
    >,
    /** The actor crtic model */
    public ac: ActorCritic<tf.Tensor>,
    /** configs for the PPO model */
    configs: DeepPartial<PPOConfigs<Observation, Action>> = {}
  ) {
    super();
    this.configs = deepMerge(this.configs, configs);

    this.env = makeEnv();
    this.obsToTensor = this.configs.obsToTensor;
    this.actionToTensor = this.configs.actionToTensor;
  }

  /**
   * Select action using the current policy network. By default selects the action by feeding the observation through the network
   * then return the argmax of the outputs
   * @param observation - observation to select action off of
   * @returns action
   */
  public act(observation: Observation): Action {
    return np.tensorLikeToNdArray(
      this.actionToTensor(this.ac.act(this.obsToTensor(observation)))
    );
  }

  public async train(trainConfigs: Partial<PPOTrainConfigs>) {
    let configs: PPOTrainConfigs = {
      vfLr: 1e-3,
      piLr: 3e-4,
      ckptFreq: 1000,
      stepsPerEpoch: 10000,
      maxEpLen: 1000,
      epochs: 50,
      trainVIters: 80,
      trainPiIters: 1,
      gamma: 0.99,
      lam: 0.97,
      clipRatio: 0.2,
      seed: 0,
      targetKl: 0.01,
      verbosity: "info",
      name: "PPO_Train",
      stepCallback: async () => {},
      epochCallback: () => {},
    };
    this.done = false;
    configs = deepMerge(configs, trainConfigs);
    log.level = configs.verbosity;

    const { clipRatio: clip_ratio, vfLr, piLr, targetKl: target_kl } = configs;
    const pi_optimizer: tf.Optimizer = tf.train.adam(piLr);
    const vf_optimizer: tf.Optimizer = tf.train.adam(vfLr);

    // TODO do some seeding things
    configs.seed += 99999;
    random.seed(configs.seed);
    // TODO: seed tensorflow if possible

    const env = this.env;
    const obs_dim = env.observationSpace.shape;
    const act_dim = env.actionSpace.shape;

    let local_steps_per_epoch = configs.stepsPerEpoch;
    if (Math.ceil(local_steps_per_epoch) !== local_steps_per_epoch) {
      configs.stepsPerEpoch = Math.ceil(local_steps_per_epoch);
      log.warn(
        `${configs.name} | Changing steps per epoch to ${configs.stepsPerEpoch} as there are 1 processes running`
      );
      local_steps_per_epoch = configs.stepsPerEpoch;
    }

    const buffer = new PPOBuffer({
      gamma: configs.gamma,
      lam: configs.lam,
      actDim: act_dim,
      obsDim: obs_dim,
      size: local_steps_per_epoch,
    });

    type pi_info = {
      approx_kl: number;
      entropy: number;
      clip_frac: any;
    };
    const compute_loss_pi = (
      data: PPOBufferComputations
    ): { loss_pi: tf.Tensor; pi_info: pi_info } => {
      const { obs, act, adv } = data;
      const logp_old = data.logp;
      return tf.tidy(() => {
        const { pi, logp_a } = this.ac.pi.apply(obs, act);

        const ratio = logp_a!.sub(logp_old).exp();
        const clip_adv = ratio
          .clipByValue(1 - clip_ratio, 1 + clip_ratio)
          .mul(adv);

        const adv_ratio = ratio.mul(adv);

        const ratio_and_clip_adv = tf.stack([adv_ratio, clip_adv]);

        const loss_pi = ratio_and_clip_adv.min(0).mean().mul(-1);

        const approx_kl = logp_old.sub(logp_a!).mean().arraySync() as number;
        const entropy = pi.entropy().mean().arraySync() as number;
        const clipped = ratio
          .greater(1 + clip_ratio)
          .logicalOr(ratio.less(1 - clip_ratio))
          .mean()
          .arraySync() as number;

        return {
          loss_pi,
          pi_info: {
            approx_kl,
            entropy,
            clip_frac: clipped,
          },
        };
      });
    };
    const compute_loss_vf = (data: PPOBufferComputations) => {
      const { obs, ret } = data;
      return this.ac.v.apply(obs).sub(ret).pow(2).mean();
    };

    const update = async () => {
      const data = await buffer.get();
      const loss_pi_old = compute_loss_pi(data).loss_pi;
      const loss_vf_old = compute_loss_vf(data);
      let kl = 0;
      let entropy = 0;
      let clip_frac = 0;
      let loss_pi_new = 0;
      let loss_vf_new = 0;

      let trained_pi_iters = configs.trainPiIters;

      for (let i = 0; i < configs.trainPiIters; i++) {
        const pi_grads = pi_optimizer.computeGradients(() => {
          const { loss_pi, pi_info } = compute_loss_pi(data);
          kl = pi_info.approx_kl;
          entropy = pi_info.entropy;
          clip_frac = pi_info.clip_frac;

          return loss_pi as tf.Scalar;
        });
        if (kl > 1.5 * target_kl) {
          log.warn(
            `${configs.name} | Early stopping at step ${i + 1}/${
              configs.trainPiIters
            } of optimizing policy due to reaching max kl`
          );
          trained_pi_iters = i + 1;
          break;
        }

        pi_optimizer.applyGradients(pi_grads.grads);
        if (i === configs.trainPiIters - 1) {
          loss_pi_new = pi_grads.value.arraySync();
        }
      }

      for (let i = 0; i < configs.trainVIters; i++) {
        if (this.stopValue) {
          this.done = true;
          return "done";
        }
        const vf_grads = vf_optimizer.computeGradients(() => {
          const loss_v = compute_loss_vf(data);
          return loss_v as tf.Scalar;
        });
        vf_optimizer.applyGradients(vf_grads.grads);
        if (i === configs.trainPiIters - 1) {
          loss_vf_new = vf_grads.value.arraySync();
        }
      }
      let delta_pi_loss = loss_pi_new - (loss_pi_old.arraySync() as number);
      let delta_vf_loss = loss_vf_new - (loss_vf_old.arraySync() as number);
      const metrics = {
        kl,
        entropy,
        delta_pi_loss,
        delta_vf_loss,
        clip_frac,
        trained_pi_iters,
      };
      return metrics;
    };

    // const start_time = process.hrtime()[0] * 1e6 + process.hrtime()[1];
    let o = env.reset();
    let ep_ret = 0;
    let ep_rets: number[] = [];
    let ep_len = 0;
    for (let epoch = 0; epoch < configs.epochs; epoch++) {
      if (this.stopValue) {
        this.done = true;
        return "done";
      }
      for (let t = 0; t < local_steps_per_epoch; t++) {
        if (this.stopValue) {
          this.done = true;
          return "done";
        }
        const { a, v, logp_a } = this.ac.step(this.obsToTensor(o));
        const action = np.tensorLikeToNdArray(this.actionToTensor(a));
        const stepInfo = env.step(action);
        const next_o = stepInfo.observation;

        const r = stepInfo.reward;
        const d = stepInfo.done;
        await configs.stepCallback({ observation: next_o, reward: r, step: t });
        ep_ret += r;
        ep_len += 1;

        buffer.store(
          np.tensorLikeToNdArray(this.obsToTensor(o)),
          np.tensorLikeToNdArray(a),
          r,
          np.tensorLikeToNdArray(v).get(0, 0),
          np.tensorLikeToNdArray(logp_a!).get(0, 0)
        );

        o = next_o;

        const timeout = ep_len === configs.maxEpLen;
        const terminal = d || timeout;
        const epoch_ended = t === local_steps_per_epoch - 1;
        if (terminal || epoch_ended) {
          if (epoch_ended && !terminal) {
            log.warn(
              `${configs.name} | Trajectory cut off by epoch at ${ep_len} steps`
            );
          }
          let v = 0;
          if (timeout || epoch_ended) {
            v = (
              this.ac.step(this.obsToTensor(o)).v.arraySync() as number[][]
            )[0][0];
          }
          buffer.finishPath(v);
          if (terminal) {
            // store ep ret and eplen stuff
            ep_rets.push(ep_ret);
          }
          o = env.reset();
          ep_ret = 0;
          ep_len = 0;
        }
      }
      // TODO save model

      // update actor critic
      await update();
      const statistics = await ct.statisticsScalar(
        np.tensorLikeToTensor(ep_rets)
      );

      configs.epochCallback({
        step: epoch,
        statistics,
      });

      ep_rets = [];
    }
  }
  async stop() {
    this.stopValue = true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  async isStop() {
    return await new Promise((resolve) => {
      while (true) {
        if (this.done) {
          resolve(true);
          return;
        }
      }
    });
  }
}
