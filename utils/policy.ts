import { randint } from "./memory";
import * as tf from "@tensorflow/tfjs";

export class Policy {
  agent;
  setAgent(agent) {
    this.agent = agent;
  }

  get metricsNames() {
    return [];
  }

  get metrics() {
    return [];
  }

  selectAction(...kwargs): number {
    throw "NotImplementedError";
  }

  getConfig() {
    return {};
  }
}

export function choice(number: number, prop: number[]): number[] {
  let addPropArray = [prop[0]];
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

export class BoltzmannQPolicy extends Policy {
  tau: number;
  clip: number[];
  constructor(tau = 1, clip = [-500, 500]) {
    super();
    this.tau = tau;
    this.clip = clip;
  }

  selectAction(qValues: number[]) {
    let nbActions = qValues.length;
    let expValues = qValues.map((qValue) => {
      let v = qValue / this.tau;
      if (v < this.clip[0]) {
        v = this.clip[0];
      } else if (this.clip[1] < v) {
        v = this.clip[1];
      }
      return Math.exp(v);
    });
    const sum = expValues.reduce((a, b) => a + b);

    let probs = expValues.map((v) => v / sum);
    let action = choice(nbActions, probs)[0];
    return action;
  }

  getConfig() {
    let config = this.getConfig();
    config["tau"] = this.tau;
    config["clip"] = this.clip;
    return config;
  }
}
export class EpsGreedyQPolicy extends Policy {
  eps = 0.1;
  constructor(eps = 0.1) {
    super();
    this.eps = eps;
  }

  selectAction(qValues: tf.Tensor) {
    const nbActions = qValues.shape[0];
    let action;
    if (Math.random() < this.eps) {
      action = randint(0, nbActions);
    } else {
      action = qValues.argMax();
    }
    return action;
  }

  getConfig() {
    const config = { eps: this.eps };
    return config;
  }
}

export class GreedyQPolicy extends Policy {
  selectAction(qValues: number[]): number {
    const action = Math.max(...qValues);
    return action;
  }
}
