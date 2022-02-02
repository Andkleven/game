import { NdArray } from "numjs";

export type Action = NdArray | number;
export type StepCallbackInput = {
  step: number;
  reward: number;
  observation: NdArray<number>;
};
export type StepCallback = (config: StepCallbackInput) => Promise<void>;
export type EpochCallbackInput = {
  step: number;
  statistics: { mean: number; max: number; min: number; std?: number };
};
export type EpochCallback = (config: EpochCallbackInput) => void;
