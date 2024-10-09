import { ViewerContext } from "./App";
import { CameraControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import * as holdEvent from "hold-event";
import React, { useContext, useRef } from "react";
import { PerspectiveCamera } from "three";
import * as THREE from "three";
import { computeT_threeworld_world } from "./WorldTransformUtils";
import { useThrottledMessageSender } from "./WebsocketFunctions";

export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const camera = useThree((state) => state.camera as PerspectiveCamera);

  const sendCameraThrottled = useThrottledMessageSender(20);

  // Helper for resetting camera poses.
  const initialCameraRef = useRef<{
    camera: PerspectiveCamera;
    lookAt: THREE.Vector3;
  } | null>(null);

  viewer.resetCameraViewRef.current = () => {
    viewer.cameraControlRef.current!.setLookAt(
      initialCameraRef.current!.camera.position.x,
      initialCameraRef.current!.camera.position.y,
      initialCameraRef.current!.camera.position.z,
      initialCameraRef.current!.lookAt.x,
      initialCameraRef.current!.lookAt.y,
      initialCameraRef.current!.lookAt.z,
      true,
    );
    viewer.cameraRef.current!.up.set(
      initialCameraRef.current!.camera.up.x,
      initialCameraRef.current!.camera.up.y,
      initialCameraRef.current!.camera.up.z,
    );
    viewer.cameraControlRef.current!.updateCameraUp();
  };

  // Callback for sending cameras.
  // It makes the code more chaotic, but we preallocate a bunch of things to
  // minimize garbage collection!
  const R_threecam_cam = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Math.PI, 0.0, 0.0),
  );
  const R_world_threeworld = new THREE.Quaternion();
  const tmpMatrix4 = new THREE.Matrix4();
  const lookAt = new THREE.Vector3();
  const R_world_camera = new THREE.Quaternion();
  const t_world_camera = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const sendCamera = React.useCallback(() => {
    const three_camera = camera;
    const camera_control = viewer.cameraControlRef.current;

    if (camera_control === null) {
      // Camera controls not yet ready, let's re-try later.
      setTimeout(sendCamera, 10);
      return;
    }

    // We put Z up to match the scene tree, and convert threejs camera convention
    // to the OpenCV one.
    const T_world_threeworld = computeT_threeworld_world(viewer).invert();
    const T_world_camera = T_world_threeworld.clone()
      .multiply(
        tmpMatrix4
          .makeRotationFromQuaternion(three_camera.quaternion)
          .setPosition(three_camera.position),
      )
      .multiply(tmpMatrix4.makeRotationFromQuaternion(R_threecam_cam));
    R_world_threeworld.setFromRotationMatrix(T_world_threeworld);

    camera_control.getTarget(lookAt).applyQuaternion(R_world_threeworld);
    const up = three_camera.up.clone().applyQuaternion(R_world_threeworld);

    //Store initial camera values
    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        camera: three_camera.clone(),
        lookAt: camera_control.getTarget(new THREE.Vector3()),
      };
    }

    T_world_camera.decompose(t_world_camera, R_world_camera, scale);

    sendCameraThrottled({
      type: "ViewerCameraMessage",
      wxyz: [
        R_world_camera.w,
        R_world_camera.x,
        R_world_camera.y,
        R_world_camera.z,
      ],
      position: t_world_camera.toArray(),
      aspect: three_camera.aspect,
      fov: (three_camera.fov * Math.PI) / 180.0,
      look_at: [lookAt.x, lookAt.y, lookAt.z],
      up_direction: [up.x, up.y, up.z],
    });
  }, [camera, sendCameraThrottled]);

  // Send camera for new connections.
  // We add a small delay to give the server time to add a callback.
  const connected = viewer.useGui((state) => state.websocketConnected);

  const initHeightOffset = parseFloat(
    new URLSearchParams(window.location.search).get("initHeightOffset") ??
      "0.05",
  );
  // Add the height offset to the initial camera position
  React.useEffect(() => {
    const cameraControls = viewer.cameraControlRef.current!;
    const lookAt = new THREE.Vector3();
    cameraControls.getTarget(lookAt);

    viewer.cameraControlRef.current!.setLookAt(
      camera.position.x,
      camera.position.y + (camera.position.y / 0.1) * initHeightOffset, // init_scale: camera.position.y / 0.1 
      camera.position.z,
      lookAt.x,
      lookAt.y + (camera.position.y / 0.1) * initHeightOffset,
      lookAt.z,
      true,
    );
    viewer.sendCameraRef.current = sendCamera;
    if (!connected) return;
    setTimeout(() => sendCamera(), 50);
  }, [connected, sendCamera]);

  // Send camera for 3D viewport changes.
  const canvas = viewer.canvasRef.current!; // R3F canvas.
  React.useEffect(() => {
    // Create a resize observer to resize the CSS canvas when the window is resized.
    const resizeObserver = new ResizeObserver(() => {
      sendCamera();
    });
    resizeObserver.observe(canvas);

    // Cleanup.
    return () => resizeObserver.disconnect();
  }, [canvas]);

  // Keyboard controls.
  React.useEffect(() => {
    const cameraControls = viewer.cameraControlRef.current!;

    const wKey = new holdEvent.KeyboardKeyHold("KeyW", 20);
    const aKey = new holdEvent.KeyboardKeyHold("KeyA", 20);
    const sKey = new holdEvent.KeyboardKeyHold("KeyS", 20);
    const dKey = new holdEvent.KeyboardKeyHold("KeyD", 20);
    const qKey = new holdEvent.KeyboardKeyHold("KeyQ", 20);
    const eKey = new holdEvent.KeyboardKeyHold("KeyE", 20);

    // TODO: these event listeners are currently never removed, even if this
    // component gets unmounted.
    aKey.addEventListener("holding", (event) => {
      cameraControls.truck(-0.0002 * event?.deltaTime, 0, true);
    });
    dKey.addEventListener("holding", (event) => {
      cameraControls.truck(0.0002 * event?.deltaTime, 0, true);
    });
    wKey.addEventListener("holding", (event) => {
      cameraControls.forward(0.0002 * event?.deltaTime, true);
    });
    sKey.addEventListener("holding", (event) => {
      cameraControls.forward(-0.0002 * event?.deltaTime, true);
    });
    qKey.addEventListener("holding", (event) => {
      cameraControls.elevate(0.0002 * event?.deltaTime, true);
    });
    eKey.addEventListener("holding", (event) => {
      cameraControls.elevate(-0.0002 * event?.deltaTime, true);
    });

    const leftKey = new holdEvent.KeyboardKeyHold("ArrowLeft", 20);
    const rightKey = new holdEvent.KeyboardKeyHold("ArrowRight", 20);
    const upKey = new holdEvent.KeyboardKeyHold("ArrowUp", 20);
    const downKey = new holdEvent.KeyboardKeyHold("ArrowDown", 20);
    leftKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    rightKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    upKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });
    downKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });

    // TODO: we currently don't remove any event listeners. This is a bit messy
    // because KeyboardKeyHold attaches listeners directly to the
    // document/window; it's unclear if we can remove these.
    return () => {
      return;
    };
  }, [CameraControls]);

  return (
    <CameraControls
      ref={viewer.cameraControlRef}
      minDistance={0.1}
      maxDistance={200.0}
      dollySpeed={0.3}
      smoothTime={0.05}
      draggingSmoothTime={0.0}
      onChange={sendCamera}
      makeDefault
    />
  );
}
