import { Environment, RenderModes } from "../../../Environments";
import { Box } from "../../../Spaces/box";
import { Discrete } from "../../../Spaces/discrete";
import nj, { NdArray } from "numjs";

export type State = NdArray<number>;
export type Observation = NdArray<number>;
export type Action = number | TensorLike;
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
}
function randomTarget() {
  return Number(
    (
      (choice(9, [0.2, 0.15, 0.15, 0, 0, 0, 0.15, 0.15, 0.2])[0] + 0.1) /
      10.0
    ).toFixed(1)
  );
}
/**
 * CartPole environment
 */
export class Custom extends Environment<
  ObservationSpace,
  ActionSpace,
  Observation,
  State,
  Action,
  Reward
> {
  public observationSpace: ObservationSpace;
  /** 0 or 1 represent applying force of -force_mag or force_mag */
  public actionSpace = new Discrete(2);
  public reachedGoal = false;
  public maxSteps = 50;
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
  constructor(configs: Partial<CartPoleConfigs> = {}) {
    super("CartPole");
    if (configs.maxEpisodeSteps) {
      this.maxEpisodeSteps = configs.maxEpisodeSteps;
    }

    this.observationSpace = new Box(0, 1, [2], "float32");
    this.actionSpace = new Discrete(2);
  }
  reset(): State {
    this.reachedGoal = false;
    this.stepsNecessary = Number(
      (
        Math.abs(this.environment["position"] - this.environment["target"]) * 10
      ).toFixed(1)
    );
    this.environment = {
      position: 0.5,
      target: randomTarget(),
    };
    this.steps = 0;
    this.steps_beyond_done = null;
    this.timestep = 0;
    return nj.array(Object.values(this.environment));
  }
  private takeStep(action: Action) {
    if (action === 0) {
      this.environment["position"] = Number(
        (this.environment["position"] - 0.1).toFixed(1)
      );
    } else if (action === 1) {
      this.environment["position"] = Number(
        (this.environment["position"] + 0.1).toFixed(1)
      );
    }
    return [this.environment.position, this.environment.target];
  }
  private setReward(): number {
    let reward = -1;
    if (this.done) {
      if (this.reachedGoal) {
        if (this.stepsNecessary === this.steps) {
          reward = 20;
        } else {
          reward = 10;
        }
      } else {
        reward = -10;
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
    if (this.maxSteps <= this.steps) {
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
    if (mode === "web") {
      console.log(`Step: ${this.step}, Reward: ${configs.rewards}`);
    } else {
      throw new Error(`${mode} is not an available render mode`);
    }
  }
}
