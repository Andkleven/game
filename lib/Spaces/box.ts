import { Action } from "./../Agent/type";
import { DataType, NdArray } from "ndarray";
import nj from "numjs";
import seedrandom from "seedrandom";
import { prng } from "../utils/random";
import * as np from "../utils/np";
import ops from "ndarray-ops";
import * as tf from "@tensorflow/tfjs";
tf.ENV.set("WEBGL_PACK", false);
import * as random from "../utils/random";
export type Shape = number[];
export interface SpaceMeta {
  discrete: boolean;
}
/**
 * Abstract Space class. T is the type of data stored
 */
export abstract class Space<T> {
  // public static Box2D = Box2D;
  public shape: Shape = [];
  constructor(
    /** Meta data about the space such as if it is a discrete space or not */
    public meta: SpaceMeta
  ) {}
  public rng: prng = seedrandom();
  abstract sample(): Action;
  abstract contains(x: T): boolean;
  seed(seed: string | undefined): void {
    this.rng = seedrandom(seed);
  }
}
export class Box extends Space<nj.NdArray<number>> {
  public low: null | NdArray;
  public high: null | NdArray;
  public rng: prng = seedrandom();
  public boundedBelow: NdArray; //tf.Tensor;
  public boundedAbove: NdArray; //tf.Tensor;
  unbounded: NdArray<any>;
  lowBounded: NdArray<any>;
  bounded: NdArray<any>;
  upperBounded: NdArray<any>;

  /**
   * @param low - lower bound for all dimensions or a list of numbers representing lower bounds for each dimension of given shape
   * @param high - upper bound for all dimensions or a list of numbers representing upper bounds for each dimension of given shape
   * @param shape - the desired box shape
   */
  constructor(
    low: null | number | nj.NdArray<number>,
    high: null | number | nj.NdArray<number>,
    public shape: Shape,
    public dtype: DataType = "float32"
  ) {
    super({ discrete: false });
    let lowerBounds: null | nj.NdArray<number> = nj.zeros(shape, dtype);
    let upperBounds: null | nj.NdArray<number> = nj.zeros(shape, dtype);
    if (typeof low === "number") {
      lowerBounds.assign(low, false);
    } else if (low === null) {
      lowerBounds = null;
    } else {
      if (!np.arrayEqual(lowerBounds.shape, low.shape))
        throw new Error(
          `low shape ${low.shape} does not match given shape ${shape}`
        );
      lowerBounds = low;
    }
    if (typeof high === "number") {
      upperBounds.assign(high, false);
    } else if (high === null) {
      upperBounds = null;
    } else {
      if (!np.arrayEqual(upperBounds.shape, high.shape))
        throw new Error(
          `high shape ${high.shape} does not match given shape ${shape}`
        );
      upperBounds = high;
    }

    this.low = lowerBounds ? np.fromNj(lowerBounds) : lowerBounds;
    this.high = upperBounds ? np.fromNj(upperBounds) : upperBounds;

    this.boundedBelow = np.zeros(shape);
    this.boundedAbove = np.zeros(shape);

    if (this.low) {
      this.boundedBelow = ops.gts(
        this.boundedBelow,
        this.low,
        Number.MIN_SAFE_INTEGER
      );
      // this.boundedBelow = tf.less(this.boundedBelow, Number.MIN_SAFE_INTEGER);
    }
    if (this.high) {
      // this.boundedAbove = tf.greater(this.boundedAbove, Number.MAX_SAFE_INTEGER);
      this.boundedAbove = ops.lts(
        this.boundedAbove,
        this.high,
        Number.MAX_SAFE_INTEGER
      );
    }

    const boundedBelowNeg = np.zeros(this.shape, "float32");
    const boundedAboveNeg = np.zeros(this.shape, "float32");

    ops.neg(boundedBelowNeg, this.boundedBelow);
    ops.addseq(boundedBelowNeg, 1);
    ops.neg(boundedAboveNeg, this.boundedAbove);
    ops.addseq(boundedAboveNeg, 1);

    this.unbounded = np.zeros(this.shape, "int8");
    this.lowBounded = np.zeros(this.shape, "int8");
    this.upperBounded = np.zeros(this.shape, "int8");
    this.bounded = np.zeros(this.shape, "int8");

    ops.and(this.unbounded, boundedBelowNeg, boundedAboveNeg);
    ops.and(this.lowBounded, this.boundedBelow, boundedAboveNeg);
    ops.and(this.upperBounded, this.boundedAbove, boundedBelowNeg);
    ops.and(this.bounded, this.boundedBelow, this.boundedAbove);
  }
  seed(seed: string | undefined): void {
    this.rng = seedrandom(seed);
  }
  sample(): Action {
    const sample = np.zeros(this.shape);
    // determine which indices are lower bounded and upper bounded
    if (this.boundedBelow === null && this.boundedAbove === null) {
      return tf.randomNormal(this.shape).arraySync();
    }

    np.set(
      sample,
      this.unbounded,
      np.fromTensorSync(tf.randomNormal(this.shape))
    );

    if (this.low) {
      const val = np.fromTensorSyncToNp(
        tf.randomGamma(this.shape, 1, 1).add(np.toTensorFromNp(this.low))
      );
      np.set(sample, this.lowBounded, val);
    }
    if (this.high) {
      np.set(
        sample,
        this.upperBounded,
        np.fromTensorSyncToNp(
          tf
            .randomGamma(this.shape, 1, 1)
            .neg()
            .add(np.toTensorFromNp(this.high))
        )
      );
    }
    if (this.low && this.high) {
      // generate uniform with shape this.shape
      const vals = np.zeros(this.shape);
      for (let i = 0; i < np.reduceMult(this.low.shape); i++) {
        const l = np.loc(this.low, i);
        const h = np.loc(this.high, i);
        vals.data[i] = random.randomVal() * (h - l) + l;
      }
      np.set(sample, this.bounded, vals);
    }
    return np.toTensorFromNp(sample).arraySync();
  }
  contains(x: NdArray): boolean {
    if (!np.arrayEqual(x.shape, this.shape)) return false;
    return true;
  }
}
