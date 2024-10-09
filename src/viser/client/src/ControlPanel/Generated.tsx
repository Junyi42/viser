import { ViewerContext } from "../App";
import { useThrottledMessageSender } from "../WebsocketFunctions";
import { GuiComponentContext } from "./GuiComponentContext";

import { Box } from "@mantine/core";
import React from "react";
import ButtonComponent from "../components/Button";
import SliderComponent from "../components/Slider";
import NumberInputComponent from "../components/NumberInput";
import TextInputComponent from "../components/TextInput";
import CheckboxComponent from "../components/Checkbox";
import Vector2Component from "../components/Vector2";
import Vector3Component from "../components/Vector3";
import DropdownComponent from "../components/Dropdown";
import RgbComponent from "../components/Rgb";
import RgbaComponent from "../components/Rgba";
import ButtonGroupComponent from "../components/ButtonGroup";
import MarkdownComponent from "../components/Markdown";
import PlotlyComponent from "../components/PlotlyComponent";
import TabGroupComponent from "../components/TabGroup";
import FolderComponent from "../components/Folder";
import MultiSliderComponent from "../components/MultiSlider";
import UploadButtonComponent from "../components/UploadButton";
import ProgressBarComponent from "../components/ProgressBar";

/** Root of generated inputs. */
export default function GeneratedGuiContainer({
  containerId,
}: {
  containerId: string;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const updateGuiProps = viewer.useGui((state) => state.updateGuiProps);
  const messageSender = useThrottledMessageSender(50);

  function setValue(id: string, value: NonNullable<unknown>) {
    updateGuiProps(id, { value: value });
    messageSender({
      type: "GuiUpdateMessage",
      id: id,
      updates: { value: value },
    });
  }
  return (
    <GuiComponentContext.Provider
      value={{
        folderDepth: 0,
        GuiContainer: GuiContainer,
        messageSender: messageSender,
        setValue: setValue,
      }}
    >
      <GuiContainer containerId={containerId} />
    </GuiComponentContext.Provider>
  );
}

function GuiContainer({ containerId }: { containerId: string }) {
  const viewer = React.useContext(ViewerContext)!;

  const guiIdSet =
    viewer.useGui((state) => state.guiIdSetFromContainerId[containerId]) ?? {};

  // Render each GUI element in this container.
  const guiIdArray = [...Object.keys(guiIdSet)];
  const guiOrderFromId = viewer!.useGui((state) => state.guiOrderFromId);
  if (guiIdSet === undefined) return null;

  let guiIdOrderPairArray = guiIdArray.map((id) => ({
    id: id,
    order: guiOrderFromId[id],
  }));
  guiIdOrderPairArray = guiIdOrderPairArray.sort((a, b) => a.order - b.order);
  const out = (
    <Box pt="xs">
      {guiIdOrderPairArray.map((pair) => (
        <GeneratedInput key={pair.id} guiId={pair.id} />
      ))}
    </Box>
  );
  return out;
}

/** A single generated GUI element. */
function GeneratedInput(props: { guiId: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const conf = viewer.useGui((state) => state.guiConfigFromId[props.guiId]);
  switch (conf.type) {
    case "GuiAddFolderMessage":
      return <FolderComponent {...conf} />;
    case "GuiAddTabGroupMessage":
      return <TabGroupComponent {...conf} />;
    case "GuiAddMarkdownMessage":
      return <MarkdownComponent {...conf} />;
    case "GuiAddPlotlyMessage":
      return <PlotlyComponent {...conf} />;
    case "GuiAddButtonMessage":
      return <ButtonComponent {...conf} />;
    case "GuiAddUploadButtonMessage":
      return <UploadButtonComponent {...conf} />;
    case "GuiAddSliderMessage":
      return <SliderComponent {...conf} />;
    case "GuiAddMultiSliderMessage":
      return <MultiSliderComponent {...conf} />;
    case "GuiAddNumberMessage":
      return <NumberInputComponent {...conf} />;
    case "GuiAddTextMessage":
      return <TextInputComponent {...conf} />;
    case "GuiAddCheckboxMessage":
      return <CheckboxComponent {...conf} />;
    case "GuiAddVector2Message":
      return <Vector2Component {...conf} />;
    case "GuiAddVector3Message":
      return <Vector3Component {...conf} />;
    case "GuiAddDropdownMessage":
      return <DropdownComponent {...conf} />;
    case "GuiAddRgbMessage":
      return <RgbComponent {...conf} />;
    case "GuiAddRgbaMessage":
      return <RgbaComponent {...conf} />;
    case "GuiAddButtonGroupMessage":
      return <ButtonGroupComponent {...conf} />;
    case "GuiAddProgressBarMessage":
      return <ProgressBarComponent {...conf} />;
    default:
      assertNeverType(conf);
  }
}

function assertNeverType(x: never): never {
  throw new Error("Unexpected object: " + (x as any).type);
}
