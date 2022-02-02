import { css } from "@emotion/react";
import { NextPage } from "next";
import dynamic from "next/dynamic";

const Effect = dynamic(() => import("../components/Hooks"));
const DefaultPlayground: NextPage = () => {
  return (
    <div
      css={css`
        display: grid;
        justify-items: center;
        align-items: center;
        width: 100%;
        height: 100%;
      `}
    >
      <div
        css={css`
          display: grid;
          justify-items: center;
          align-items: center;
          width: 100%;
          height: 100%;
        `}
      >
        <Effect />
      </div>
    </div>
  );
};

export default DefaultPlayground;
