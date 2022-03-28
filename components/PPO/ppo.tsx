import { css } from "@emotion/react";

import { Custom } from "../../lib/Environments/examples/custom/index";
import { seed } from "../../lib/utils/random";
import { MLPActorCritic } from "../../lib/Models/ac";
import { PPO, PPOTrainConfigs } from "../../lib/Algos/ppo/index";
import { Fragment, useCallback, useRef, useState } from "react";
import { MdEmojiPeople } from "react-icons/md";
import { BiTargetLock } from "react-icons/bi";
import { AiOutlineStop } from "react-icons/ai";
import {
  EpochCallback,
  EpochCallbackInput,
  StepCallback,
} from "../../lib/Agent/type";
import HelpText from "../HelpText";
import { Line } from "react-chartjs-2";
import { Slider, Button, NumberInput, Paper } from "@mantine/core";
import { useForm } from "@mantine/hooks";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const borders = new Array(11).fill(0);

type Config = Omit<
  PPOTrainConfigs,
  "stepCallback" | "epochCallback" | "seed" | "ckptFreq" | "verbosity" | "name"
>;
const makeEnv = () => {
  return new Custom();
};
const env = makeEnv();

const Ppo = () => {
  const agent = useRef(
    new PPO(
      makeEnv,
      new MLPActorCritic(env.observationSpace, env.actionSpace, [24, 48]),
      {
        actionToTensor: (action) => {
          return action.argMax(1);
        },
      }
    )
  );
  const [observation, setObservation] = useState<{
    people: number;
    target: number;
  }>({
    people: 0.5,
    target: 0.7,
  });
  const [epochs, setEpochs] = useState<EpochCallbackInput[]>([]);
  const form = useForm<Config>({
    initialValues: {
      epochs: 50,
      gamma: 0.99,
      stepsPerEpoch: 1000,
      clipRatio: 0.2,
      piLr: 1e-3,
      vfLr: 3e-4,
      trainVIters: 80,
      trainPiIters: 1,
      lam: 0.97,
      maxEpLen: 1000,
      targetKl: 0.01,
    },
  });
  const runModel = useCallback(async (configs: Config) => {
    seed(0);
    setEpochs([]);
    let obs = env.reset();
    setObservation({
      people: obs.get(0),
      target: obs.get(1),
    });
    const stepCallback: StepCallback = async ({ observation }) => {
      setObservation({
        people: observation.get(0),
        target: observation.get(1),
      });
      await new Promise((resolve) => setTimeout(resolve, 5));
    };
    const epochCallback: EpochCallback = (config) => {
      setEpochs((prevState) => {
        return [config, ...prevState];
      });
    };
    const done = await agent.current.train({
      ...configs,
      stepCallback,
      epochCallback,
    });
    return done;
  }, []);

  const data = {
    labels: epochs.map(({ step }) => step.toString()),
    datasets: [
      {
        label: "Max",
        data: epochs.map(({ statistics }) => statistics["max"]),
        borderColor: "rgb(99, 255, 132)",
        backgroundColor: "rgba(99, 255, 132, 0.5)",
      },
      {
        label: "Min",
        data: epochs.map(({ statistics }) => statistics["min"]),
        borderColor: "rgb(255, 99, 132)",
        backgroundColor: "rgba(255, 99, 132, 0.5)",
      },
      {
        label: "Mean",
        data: epochs.map(({ statistics }) => statistics["mean"]),
        borderColor: "rgb(99, 132, 255)",
        backgroundColor: "rgba(99, 132, 255, 0.5)",
      },
    ],
  };

  const showChart = epochs.length > 0;

  return (
    <div
      css={css`
        width: 100%;
        height: 100%;
        display: grid;
        align-items: start;
        justify-content: end;
        padding: 20px;
        grid-gap: 5px;
        grid-template-columns: 1fr 1fr;
        grid-template-areas: ${showChart
          ? '"statistics statistics" "environment form"'
          : '"environment form"'};
      `}
    >
      {showChart ? (
        <div
          css={css`
            grid-area: statistics;
          `}
        >
          <Line
            data={data}
            options={{
              responsive: true,
              plugins: {
                title: {
                  display: true,
                  text: "Epochs",
                },
              },
            }}
          />
        </div>
      ) : null}
      <div
        css={css`
          grid-area: environment;
          display: grid;
          padding: 20px;
          grid-gap: 5px;
        `}
      >
        {borders.map((_, index) => {
          return (
            <div
              key={index + "borders"}
              css={css`
                text-align: center;
                border: 2px solid;
                height: 70px;
                width: 70px;
              `}
            >
              {index}
              <div>
                {[0, 10].includes(index) && <AiOutlineStop size={25} />}
                {observation.people * 10 === index && (
                  <MdEmojiPeople size={25} />
                )}
                {observation.target * 10 === index && (
                  <BiTargetLock size={25} />
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div
        css={css`
          grid-area: form;
          display: grid;
          align-items: center;
          padding: 20px;
        `}
      >
        <Paper>
          <form
            css={css`
              display: grid;
              align-items: center;
              grid-gap: 15px;
            `}
            onSubmit={form.onSubmit(async (values) => {
              await agent.current.stop();
              await agent.current.isStop();
              agent.current = new PPO(
                makeEnv,
                new MLPActorCritic(
                  env.observationSpace,
                  env.actionSpace,
                  [24, 48]
                ),
                {
                  actionToTensor: (action) => {
                    return action.argMax(1);
                  },
                }
              );
              runModel(values);
            })}
          >
            <HelpText
              text={
                "Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch."
              }
            >
              <NumberInput
                required
                label="Steps Per Epoch"
                {...form.getInputProps("stepsPerEpoch")}
              />
            </HelpText>
            <HelpText
              text={
                "Number of Epochs is the number of passes through the experience buffer during gradient descent. The larger the batch size, the larger it is acceptable to make this. Decreasing this will ensure more stable updates, at the cost of slower learning"
              }
            >
              <NumberInput
                required
                label="Number of Epochs"
                {...form.getInputProps("epochs")}
              />
            </HelpText>
            <div>
              <label>Gamma</label>
              <HelpText
                text={
                  "Gamma corresponds to the discount factor for future rewards. This can be thought of as how far into the future the agent should care about possible rewards. In situations when the agent should be acting in the present in order to prepare for rewards in the distant future, this value should be large. In cases when rewards are more immediate, it can be smaller. \n \n Typical Range: 0.8 - 0.995"
                }
              >
                <Slider
                  labelAlwaysOn
                  label={(value) => value.toFixed(2)}
                  {...form.getInputProps("gamma")}
                  step={0.01}
                  min={0}
                  max={1}
                />
              </HelpText>
            </div>
            <div>
              <label>Clip Ratio</label>
              <HelpText
                text={
                  "Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy while still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn’t help on the objective anymore. (Usually small, 0.1 to 0.3.) Typically denoted by epsilon"
                }
              >
                <Slider
                  labelAlwaysOn
                  label={(value) => value.toFixed(1)}
                  {...form.getInputProps("clipRatio")}
                  step={0.1}
                  min={0.1}
                  max={0.5}
                />
              </HelpText>
            </div>
            <div>
              <label>pi_lr</label>
              <HelpText text={"Learning rate for policy optimizer."}>
                <Slider
                  labelAlwaysOn
                  label={(value) => value.toFixed(6)}
                  {...form.getInputProps("piLr")}
                  step={1e-6}
                  max={1e-2}
                  min={1e-6}
                />
              </HelpText>
            </div>
            <div>
              <label>vf_lr</label>
              <HelpText text={"Learning rate for value function optimizer."}>
                <Slider
                  labelAlwaysOn
                  label={(value) => value.toFixed(6)}
                  {...form.getInputProps("vfLr")}
                  step={1e-6}
                  max={1e-2}
                  min={1e-6}
                />
              </HelpText>
            </div>
            <HelpText
              text={
                "Number of gradient descent steps to take on value function per epoch."
              }
            >
              <NumberInput
                required
                label="train_v_iters"
                {...form.getInputProps("trainVIters")}
              />
            </HelpText>
            <HelpText
              text={
                "Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)"
              }
            >
              <NumberInput
                required
                label="train_pi_iters"
                {...form.getInputProps("trainPiIters")}
              />
            </HelpText>
            <HelpText
              text={"Maximum length of trajectory / episode / rollout."}
            >
              <NumberInput
                required
                label="MaxEpLen"
                {...form.getInputProps("maxEpLen")}
              />
            </HelpText>
            <div>
              <label>targetKl</label>
              <HelpText
                text={
                  "Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)"
                }
              >
                <Slider
                  labelAlwaysOn
                  label={(value) => value.toFixed(3)}
                  {...form.getInputProps("targetKl")}
                  step={0.001}
                  max={0.2}
                  min={0.0}
                />
              </HelpText>
            </div>
            <Button
              css={css`
                width: 100%;
              `}
              onClick={() => form.reset()}
              type="button"
            >
              Reset
            </Button>
            <Button
              css={css`
                width: 100%;
              `}
              type="submit"
            >
              Apply
            </Button>
          </form>
        </Paper>
      </div>
    </div>
  );
};

export default Ppo;
