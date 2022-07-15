import { Environment, RenderModes } from "../../../Environments";
import { Box } from "../../../Spaces/box";
import { Discrete } from "../../../Spaces/discrete";
import nj, { NdArray } from "numjs";
import { Action } from "../../../Agent/type";
import { tensorLikeToNdArray } from "../../../utils/np";

export type State = NdArray<number>;
export type Observation = NdArray<number>;
export type ActionSpace = Discrete;
export type ObservationSpace = Box;
export type Reward = number;
export function choice(number: number, prop: number[]): number[] {
  const addPropArray = [prop[0]];
  prop.reduce((a, b) => {
    addPropArray.push(a + b);
    return a + b;
  });

  return Array.apply(null, Array(number)).map((_) => {
    const random = Math.random();
    let returnValue;
    for (let index = 0; index < addPropArray.length; index++) {
      const prop = addPropArray[index];
      if (random <= prop && prop !== 0) {
        returnValue = index;
        break;
      }
    }
    return returnValue;
  });
}
export interface CartPoleConfigs {
  maxEpisodeSteps: number;
  rewardStep: number;
  rewardTargetReached: number;
  rewardTargetNotReached: number;
  reachedTargetFirstTry: number;
}
function randomTarget() {
  return Number(
    (
      (choice(1, [0.2, 0.15, 0.15, 0, 0, 0, 0.15, 0.15, 0.2])[0] + 1) /
      10.0
    ).toFixed(1)
  );
}
/**
 * CartPole environment
 */
export class SquareGame extends Environment<
  ObservationSpace,
  ActionSpace,
  Observation,
  State,
  Action,
  Reward
> {
  public observationSpace = new Box(0, 1, [2], "float32");
  /** 0 or 1 represent applying force of -force_mag or force_mag */
  public actionSpace = new Discrete(2);
  public reachedGoal = false;
  public done = false;
  public steps = 0;
  public steps_beyond_done: null | number = null;
  public timestep = 0;
  public maxEpisodeSteps = 500;
  public globalTimestep = 0;
  public environment = {
    position: 0.5,
    target: randomTarget(),
  };
  public stepsNecessary = Number(
    (
      Math.abs(this.environment["position"] - this.environment["target"]) * 10
    ).toFixed(1)
  );
  public state: NdArray = nj.array(Object.values(this.environment));
  private rewardStep: number = 0;
  private rewardTargetReached: number = 0;
  private rewardTargetNotReached: number = 0;
  constructor({
    maxEpisodeSteps,
    rewardStep,
    rewardTargetNotReached,
    rewardTargetReached,
  }: CartPoleConfigs) {
    super("CartPole");
    if (maxEpisodeSteps) {
      this.maxEpisodeSteps = maxEpisodeSteps;
    }
    this.rewardStep = rewardStep;
    this.rewardTargetNotReached = rewardTargetNotReached;
    this.rewardTargetReached = rewardTargetReached;
  }
  reset(): State {
    this.reachedGoal = false;
    this.environment = {
      position: 0.5,
      target: randomTarget(),
    };
    this.stepsNecessary = Number(
      (
        Math.abs(this.environment["position"] - this.environment["target"]) * 10
      ).toFixed(1)
    );
    this.steps = 0;
    this.steps_beyond_done = null;
    this.timestep = 0;
    return nj.array([this.environment.position, this.environment.target]);
  }
  private takeStep(action: Action) {
    const actionArray = tensorLikeToNdArray(action);
    if (actionArray.get(0) === 0) {
      this.environment["position"] = Number(
        (this.environment["position"] - 0.1).toFixed(1)
      );
    } else if (actionArray.get(0) === 1) {
      this.environment["position"] = Number(
        (this.environment["position"] + 0.1).toFixed(1)
      );
    }
    return [this.environment.position, this.environment.target];
  }
  private setReward(): number {
    let reward = this.rewardStep;
    if (this.done) {
      if (this.reachedGoal) {
        if (this.stepsNecessary === this.steps) {
          reward = 20;
        } else {
          reward = this.rewardTargetReached;
        }
      } else {
        reward = this.rewardTargetNotReached;
      }
    }
    return reward;
  }

  private isDone(): boolean {
    let done = false;
    if (
      this.environment["position"] === 0.0 ||
      this.environment["position"] === 1.0
    ) {
      done = true;
    }
    if (this.environment["position"] === this.environment["target"]) {
      this.reachedGoal = true;
      done = true;
    }
    if (this.maxEpisodeSteps <= this.steps) {
      done = true;
    }
    return done;
  }
  step(action: Action) {
    this.steps += 1;
    const observation = this.takeStep(action);
    const done = this.isDone();
    this.done = done;
    const reward = this.setReward();
    return {
      observation: nj.array(observation),
      reward,
      done,
    };
  }

  async render(
    mode: RenderModes,
    configs: { fps: number; episode?: number; rewards?: number } = { fps: 60 }
  ): Promise<void> {
    // if (mode === "web") {
    //   console.log(`Step: ${this.step}, Reward: ${configs.rewards}`);
    // } else {
    //   throw new Error(`${mode} is not an available render mode`);
    // }
  }
}
