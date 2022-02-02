import { css } from "@emotion/react";
import { Custom } from "../../lib/Environments/examples/custom/index";
import { seed } from "../../lib/utils/random";
import { MLPActorCritic } from "../../lib/Models/ac";
import { VPG } from "../../lib/Algos/vpg/index";
import { Fragment, useEffect, useState } from "react";
import { MdEmojiPeople } from "react-icons/md";
import { BiTargetLock } from "react-icons/bi";
import {
  EpochCallback,
  EpochCallbackInput,
  StepCallback,
} from "../../lib/Agent/type";
const borders = new Array(9).fill(0);

const Vpg = () => {
  const [observation, setObservation] = useState<{
    people: number;
    target: number;
  }>({
    people: 0,
    target: 0,
  });
  const [epochs, setEpochs] = useState<EpochCallbackInput[]>([]);
  useEffect(() => {
    seed(0);
    const makeEnv = () => {
      return new Custom();
    };
    const env = makeEnv();
    let obs = env.reset();
    setObservation({
      people: obs.get(0),
      target: obs.get(1),
    });
    const ac = new MLPActorCritic(
      env.observationSpace,
      env.actionSpace,
      [24, 48]
    );
    const vpg = new VPG(makeEnv, ac, {
      actionToTensor: (action) => {
        return action.argMax(1);
      },
    });
    const stepCallback: StepCallback = async (config) => {
      setObservation({
        people: config.observation.get(0),
        target: config.observation.get(1),
      });
      await new Promise((resolve) => setTimeout(resolve, 5));
    };
    const epochCallback: EpochCallback = (config) => {
      setEpochs((prevState) => {
        return [config, ...prevState];
      });
    };
    vpg.train({
      steps_per_epoch: 1000,
      epochs: 200,
      train_pi_iters: 10,
      train_v_iters: 80,
      stepCallback,
      epochCallback,
    });
  }, []);
  return (
    <div
      css={css`
        width: 100%;
        height: 100%;
        display: grid;
        justify-content: center;
        align-items: start;
        padding: 20px;
        grid-gap: 5px;
        grid-template-columns: 1fr 1fr;
      `}
    >
      <div>
        {epochs.map(({ statistics, step }, index) => (
          <Fragment key={[index, step, "Epoch"].join("-")}>
            <h3>Epoch: {step + 1}</h3>
            {Object.keys(statistics).map((key, index) => {
              return (
                <div key={["statistics", key, index, step].join("-")}>
                  {key}: {statistics[key]}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <div
        css={css`
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
              {index + 1}
              <div>
                {observation.people * 10 === index + 1 && (
                  <MdEmojiPeople size={25} />
                )}
                {observation.target * 10 === index + 1 && (
                  <BiTargetLock size={25} />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Vpg;
