import { useCursor } from "@react-three/drei";
import { createPortal, useFrame } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { ViewerContext } from "./App";
import { useThrottledMessageSender } from "./WebsocketFunctions";
import { Html } from "@react-three/drei";
import { immerable } from "immer";
import { useSceneTreeState } from "./SceneTreeState";
import { ErrorBoundary } from "react-error-boundary";
import { rayToViserCoords } from "./WorldTransformUtils";
import { HoverableContext } from "./ThreeAssets";
import { opencvXyFromPointerXy } from "./ClickUtils";

export type MakeObject<T extends THREE.Object3D = THREE.Object3D> = (
  ref: React.Ref<T>,
) => React.ReactNode;

/** Scenes will consist of nodes, which form a tree. */
export class SceneNode<T extends THREE.Object3D = THREE.Object3D> {
  [immerable] = true;

  public children: string[];
  public clickable: boolean;

  constructor(
    public readonly name: string,
    public readonly makeObject: MakeObject<T>,
    public readonly cleanup?: () => void,
    /** unmountWhenInvisible is used to unmount <Html /> components when they
     * should be hidden.
     *
     * https://github.com/pmndrs/drei/issues/1323
     */
    public readonly unmountWhenInvisible?: boolean,
    public readonly everyFrameCallback?: () => void,
    /** For click events on instanced nodes, like batched axes, we want to keep track of which. */
    public readonly computeClickInstanceIndexFromInstanceId?: (
      instanceId: number | undefined,
    ) => number | null,
  ) {
    this.children = [];
    this.clickable = false;
  }
}

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

function SceneNodeThreeChildren(props: {
  name: string;
  parent: THREE.Object3D;
}) {
  const viewer = React.useContext(ViewerContext)!;

  const [children, setChildren] = React.useState<string[]>(
    viewer.useSceneTree.getState().nodeFromName[props.name]?.children ?? [],
  );

  React.useEffect(() => {
    let updateQueued = false;
    return viewer.useSceneTree.subscribe((state) => {
      // Do nothing if an update is already queued.
      if (updateQueued) return;

      // Do nothing if children haven't changed.
      const newChildren = state.nodeFromName[props.name]?.children;
      if (
        newChildren === undefined ||
        newChildren === children || // Note that this won't check for elementwise equality!
        (newChildren.length === 0 && children.length == 0)
      )
        return;

      // Queue a (throttled) children update.
      updateQueued = true;
      setTimeout(
        () => {
          updateQueued = false;
          const newChildren =
            viewer.useSceneTree.getState().nodeFromName[props.name]!.children!;
          setChildren(newChildren);
        },
        // Throttle more when we have a lot of children...
        newChildren.length <= 16 ? 10 : newChildren.length <= 128 ? 50 : 200,
      );
    });
  }, []);

  // Create a group of children inside of the parent object.
  return createPortal(
    <group>
      {children &&
        children.map((child_id) => (
          <SceneNodeThreeObject
            key={child_id}
            name={child_id}
            parent={props.parent}
          />
        ))}
      <SceneNodeLabel name={props.name} />
    </group>,
    props.parent,
  );
}

/** Component for updating attributes of a scene node. */
function SceneNodeLabel(props: { name: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const labelVisible = viewer.useSceneTree(
    (state) => state.labelVisibleFromName[props.name],
  );
  return labelVisible ? (
    <Html>
      <span
        style={{
          backgroundColor: "rgba(240, 240, 240, 0.9)",
          borderRadius: "0.2rem",
          userSelect: "none",
          padding: "0.1em 0.2em",
        }}
      >
        {props.name}
      </span>
    </Html>
  ) : null;
}

/** Component containing the three.js object and children for a particular scene node. */
export function SceneNodeThreeObject(props: {
  name: string;
  parent: THREE.Object3D | null;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const makeObject = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.makeObject,
  );
  const unmountWhenInvisible = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.unmountWhenInvisible,
  );
  const everyFrameCallback = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.everyFrameCallback,
  );
  const computeClickInstanceIndexFromInstanceId = viewer.useSceneTree(
    (state) =>
      state.nodeFromName[props.name]?.computeClickInstanceIndexFromInstanceId,
  );
  const [unmount, setUnmount] = React.useState(false);
  const clickable =
    viewer.useSceneTree((state) => state.nodeFromName[props.name]?.clickable) ??
    false;
  const [obj, setRef] = React.useState<THREE.Object3D | null>(null);

  // Update global registry of node objects.
  // This is used for updating bone transforms in skinned meshes.
  React.useEffect(() => {
    if (obj !== null) viewer.nodeRefFromName.current[props.name] = obj;
  }, [obj]);

  // Create object + children.
  //
  // For not-fully-understood reasons, wrapping makeObject with useMemo() fixes
  // stability issues (eg breaking runtime errors) associated with
  // PivotControls.
  const objNode = React.useMemo(() => {
    if (makeObject === undefined) return null;

    // Pose will need to be updated.
    const attrs = viewer.nodeAttributesFromName.current;
    if (!(props.name in attrs)) {
      attrs[props.name] = {};
    }
    attrs[props.name]!.poseUpdateState = "needsUpdate";

    return makeObject(setRef);
  }, [makeObject]);
  const children =
    obj === null ? null : (
      <SceneNodeThreeChildren name={props.name} parent={obj} />
    );

  // Helper for transient visibility checks. Checks the .visible attribute of
  // both this object and ancestors.
  //
  // This is used for (1) suppressing click events and (2) unmounting when
  // unmountWhenInvisible is true. The latter is used for <Html /> components.
  function isDisplayed() {
    // We avoid checking obj.visible because obj may be unmounted when
    // unmountWhenInvisible=true.
    const attrs = viewer.nodeAttributesFromName.current[props.name];
    const visibility =
      (attrs?.overrideVisibility === undefined
        ? attrs?.visibility
        : attrs.overrideVisibility) ?? true;
    if (visibility === false) return false;
    if (props.parent === null) return true;

    // Check visibility of parents + ancestors.
    let visible = props.parent.visible;
    if (visible) {
      props.parent.traverseAncestors((ancestor) => {
        visible = visible && ancestor.visible;
      });
    }
    return visible;
  }

  // Pose needs to be updated whenever component is remounted.
  React.useEffect(() => {
    const attrs = viewer.nodeAttributesFromName.current[props.name];
    if (attrs !== undefined) attrs.poseUpdateState = "needsUpdate";
  });

  // Update attributes on a per-frame basis. Currently does redundant work,
  // although this shouldn't be a bottleneck.
  useFrame(
    () => {
      const attrs = viewer.nodeAttributesFromName.current[props.name];
      everyFrameCallback && everyFrameCallback();

      // Unmount when invisible.
      // Examples: <Html /> components, PivotControls.
      //
      // This is a workaround for situations where just setting `visible` doesn't
      // work (like <Html />), or to prevent invisible elements from being
      // interacted with (<PivotControls />).
      //
      // https://github.com/pmndrs/drei/issues/1323
      if (unmountWhenInvisible) {
        const displayed = isDisplayed();
        if (displayed && unmount) {
          if (obj !== null) obj.visible = false;
          setUnmount(false);
        }
        if (!displayed && !unmount) {
          setUnmount(true);
        }
      }

      if (obj === null) return;
      if (attrs === undefined) return;

      const visibility =
        (attrs?.overrideVisibility === undefined
          ? attrs?.visibility
          : attrs.overrideVisibility) ?? true;
      obj.visible = visibility;

      if (attrs.poseUpdateState == "needsUpdate") {
        attrs.poseUpdateState = "updated";
        const wxyz = attrs.wxyz;
        if (wxyz !== undefined) {
          obj.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
        }
        const position = attrs.position;
        if (position !== undefined) {
          obj.position.set(position[0], position[1], position[2]);
        }

        // Update matrices if necessary. This is necessary for PivotControls.
        if (!obj.matrixAutoUpdate) obj.updateMatrix();
        if (!obj.matrixWorldAutoUpdate) obj.updateMatrixWorld();
      }
    },
    // Other useFrame hooks may depend on transforms + visibility. So it's best
    // to call this hook early.
    -10000,
  );

  // Clicking logic.
  const sendClicksThrottled = useThrottledMessageSender(50);
  const [hovered, setHovered] = React.useState(false);
  useCursor(hovered);
  const hoveredRef = React.useRef(false);
  if (!clickable && hovered) setHovered(false);

  const dragInfo = React.useRef({
    dragging: false,
    startClientX: 0,
    startClientY: 0,
  });

  if (objNode === undefined || unmount) {
    return <>{children}</>;
  } else if (clickable) {
    return (
      <>
        <ErrorBoundary
          fallbackRender={() => {
            // This sometimes (but very rarely) catches a race condition when
            // we remove scene nodes. I would guess it's related to portaling,
            // but the issue is unnoticeable with ErrorBoundary in-place so not
            // debugging further for now...
            console.error(
              "There was an error rendering a scene node object:",
              objNode,
            );
            return null;
          }}
        >
          <group
            // Instead of using onClick, we use onPointerDown/Move/Up to check mouse drag,
            // and only send a click if the mouse hasn't moved between the down and up events.
            //  - onPointerDown resets the click state (dragged = false)
            //  - onPointerMove, if triggered, sets dragged = true
            //  - onPointerUp, if triggered, sends a click if dragged = false.
            // Note: It would be cool to have dragged actions too...
            onPointerDown={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              const state = dragInfo.current;
              const canvasBbox =
                viewer.canvasRef.current!.getBoundingClientRect();
              state.startClientX = e.clientX - canvasBbox.left;
              state.startClientY = e.clientY - canvasBbox.top;
              state.dragging = false;
            }}
            onPointerMove={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              const state = dragInfo.current;
              const canvasBbox =
                viewer.canvasRef.current!.getBoundingClientRect();
              const deltaX = e.clientX - canvasBbox.left - state.startClientX;
              const deltaY = e.clientY - canvasBbox.top - state.startClientY;
              // Minimum motion.
              if (Math.abs(deltaX) <= 3 && Math.abs(deltaY) <= 3) return;
              state.dragging = true;
            }}
            onPointerUp={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              const state = dragInfo.current;
              if (state.dragging) return;
              // Convert ray to viser coordinates.
              const ray = rayToViserCoords(viewer, e.ray);

              // Send OpenCV image coordinates to the server (normalized).
              const canvasBbox =
                viewer.canvasRef.current!.getBoundingClientRect();
              const mouseVectorOpenCV = opencvXyFromPointerXy(viewer, [
                e.clientX - canvasBbox.left,
                e.clientY - canvasBbox.top,
              ]);

              sendClicksThrottled({
                type: "SceneNodeClickMessage",
                name: props.name,
                instance_index:
                  computeClickInstanceIndexFromInstanceId === undefined
                    ? null
                    : computeClickInstanceIndexFromInstanceId(e.instanceId),
                // Note that the threejs up is +Y, but we expose a +Z up.
                ray_origin: [ray.origin.x, ray.origin.y, ray.origin.z],
                ray_direction: [
                  ray.direction.x,
                  ray.direction.y,
                  ray.direction.z,
                ],
                screen_pos: [mouseVectorOpenCV.x, mouseVectorOpenCV.y],
              });
            }}
            onPointerOver={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              setHovered(true);
              hoveredRef.current = true;
            }}
            onPointerOut={() => {
              if (!isDisplayed()) return;
              setHovered(false);
              hoveredRef.current = false;
            }}
          >
            <HoverableContext.Provider value={hoveredRef}>
              {objNode}
            </HoverableContext.Provider>
          </group>
          {children}
        </ErrorBoundary>
      </>
    );
  } else {
    return (
      <>
        {/* This <group /> does nothing, but switching between clickable vs not
        causes strange transform behavior without it. */}
        <group>{objNode}</group>
        {children}
      </>
    );
  }
}
