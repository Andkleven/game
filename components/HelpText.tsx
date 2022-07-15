import { css } from "@emotion/react";
import {
  Popper,
  Center,
  useMantineTheme,
  ActionIcon,
  Paper,
} from "@mantine/core";
import { ReactNode, useState } from "react";
import { BsQuestionLg } from "react-icons/bs";

export default function HelpText({
  text,
  children,
}: {
  text: String;
  children: ReactNode;
}) {
  const [referenceElement, setReferenceElement] = useState();
  const [visible, setVisible] = useState(false);
  const theme = useMantineTheme();

  return (
    <div
      css={css`
        display: grid;
        grid-template-columns: auto 20px;
        align-items: center;
      `}
    >
      {children}
      <ActionIcon
        ref={setReferenceElement as any}
        onClick={() => setVisible((m) => !m)}
      >
        <BsQuestionLg />
      </ActionIcon>
      <Popper
        position="bottom"
        placement="end"
        gutter={5}
        arrowSize={5}
        withArrow
        mounted={visible}
        referenceElement={referenceElement as any}
        transition="pop-top-left"
        transitionDuration={200}
        arrowStyle={{
          backgroundColor:
            theme.colorScheme === "dark"
              ? theme.colors.dark[5]
              : theme.colors.gray[1],
        }}
      >
        <Paper
          style={{
            backgroundColor:
              theme.colorScheme === "dark"
                ? theme.colors.dark[5]
                : theme.colors.gray[1],
          }}
        >
          <Center
            style={{ maxWidth: 500, padding: "10px", whiteSpace: "pre-line" }}
          >
            {text}
          </Center>
        </Paper>
      </Popper>
    </div>
  );
}
