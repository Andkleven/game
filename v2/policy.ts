export class EpsilonGreedyPolicy {
  start: number;
  end: number;
  decay: number;
  constructor(start: number, end: number, decay: number) {
    this.start = start;
    this.end = end;
    this.decay = decay;
  }
  getExplorationRate(currentStep: number) {
    return (
      this.end +
      (this.start - this.end) * Math.exp(-1 * currentStep * this.decay)
    );
  }
}
