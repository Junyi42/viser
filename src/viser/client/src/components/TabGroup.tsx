import * as React from "react";
import { GuiAddTabGroupMessage } from "../WebsocketMessages";
import { Tabs } from "@mantine/core";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { htmlIconWrapper } from "./ComponentStyles.css";

export default function TabGroupComponent({
  tab_labels,
  tab_icons_html,
  tab_container_ids,
  visible,
}: GuiAddTabGroupMessage) {
  const { GuiContainer } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return (
    <Tabs radius="xs" defaultValue={"0"} style={{ marginTop: "-0.55em" }}>
      <Tabs.List>
        {tab_labels.map((label, index) => (
          <Tabs.Tab
            value={index.toString()}
            key={index}
            styles={{
              tabSection: { marginRight: "0.5em" },
              tab: { padding: "0.75em" },
            }}
            leftSection={
              tab_icons_html[index] === null ? undefined : (
                <div
                  className={htmlIconWrapper}
                  dangerouslySetInnerHTML={{ __html: tab_icons_html[index]! }}
                />
              )
            }
          >
            {label}
          </Tabs.Tab>
        ))}
      </Tabs.List>
      {tab_container_ids.map((containerId, index) => (
        <Tabs.Panel value={index.toString()} key={containerId}>
          <GuiContainer containerId={containerId} />
        </Tabs.Panel>
      ))}
    </Tabs>
  );
}
