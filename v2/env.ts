import { getRandomInt } from "./tools";
import { choice } from "../utils/policy";

export class Env {
  observation: number[] = [];
  action: number | undefined;
  maxSteps = 50;
  done = false;
  steps = 0;
  numberOfAction = 0;
  get getNumberOfObservation() {
    return this.observation.length;
  }
  get getNumberOfAction() {
    return this.numberOfAction;
  }
  reset(): number[] {
    this.steps = 0;
    this.done = false;
    return this.resetEnvironment();
  }
  step(action: number): [observation: number[], reward: number, done: boolean] {
    this.steps += 1;
    const observation = this.takeStep(action);
    const done = this.isDone();
    this.done = done;
    const reward = this.setReward();
    return [observation, reward, done];
  }
  resetEnvironment(): number[] {
    throw new Error("NotImplementedError");
  }
  takeStep(action: number): number[] {
    throw new Error("NotImplementedError");
  }
  setReward(): number {
    throw new Error("NotImplementedError");
  }
  isDone(): boolean {
    throw new Error("NotImplementedError");
  }
  sample() {
    return getRandomInt(this.numberOfAction);
  }
}

function randomTarget() {
  return Number(
    (
      (choice(9, [0.2, 0.15, 0.15, 0, 0, 0, 0.15, 0.15, 0.2])[0] + 0.1) /
      10.0
    ).toFixed(1)
  );
}
export class DQNEnv extends Env {
  maxSteps = 20;
  observationShape = 2;
  numberOfAction = 2;
  reachedGoal = false;
  stepsNecessary = 0;
  environment = {
    position: 0.5,
    target: randomTarget(),
  };
  observation = [this.environment.position, this.environment.target];

  constructor() {
    super();
    this.reset();
  }

  resetEnvironment() {
    this.environment = {
      position: 0.5,
      target: randomTarget(),
    };
    this.reachedGoal = false;
    this.stepsNecessary = Number(
      (
        Math.abs(this.environment["position"] - this.environment["target"]) * 10
      ).toFixed(1)
    );
    return [this.environment["position"], this.environment["target"]];
  }
  takeStep(action: number) {
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
  setReward(): number {
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

  isDone(): boolean {
    let done = false;
    this.environment["position"] = this.environment["position"];
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
}
