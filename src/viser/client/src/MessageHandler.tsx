import { CatmullRomLine, CubicBezierLine, Grid, Html } from "@react-three/drei";
import { useContextBridge } from "its-fine";
import { notifications } from "@mantine/notifications";

import React, { useContext } from "react";
import * as THREE from "three";
import { TextureLoader } from "three";

import { ViewerContext } from "./App";
import { SceneNode } from "./SceneTree";
import {
  CameraFrustum,
  CoordinateFrame,
  InstancedAxes,
  GlbAsset,
  OutlinesIfHovered,
  PointCloud,
} from "./ThreeAssets";
import {
  FileTransferPart,
  FileTransferStart,
  Message,
} from "./WebsocketMessages";
import { PivotControls } from "@react-three/drei";
import { isTexture, makeThrottledMessageSender } from "./WebsocketFunctions";
import { isGuiConfig } from "./ControlPanel/GuiState";
import { useFrame } from "@react-three/fiber";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { Paper, Progress } from "@mantine/core";
import { IconCheck } from "@tabler/icons-react";
import { computeT_threeworld_world } from "./WorldTransformUtils";
import { SplatObject } from "./Splatting/GaussianSplats";

/** Convert raw RGB color buffers to linear color buffers. **/
function threeColorBufferFromUint8Buffer(colors: ArrayBuffer) {
  return new THREE.Float32BufferAttribute(
    new Float32Array(new Uint8Array(colors)).map((value) => {
      value = value / 255.0;
      if (value <= 0.04045) {
        return value / 12.92;
      } else {
        return Math.pow((value + 0.055) / 1.055, 2.4);
      }
    }),
    3,
  );
}

/** Returns a handler for all incoming messages. */
function useMessageHandler() {
  const viewer = useContext(ViewerContext)!;
  const ContextBridge = useContextBridge();

  // We could reduce the redundancy here if we wanted to.
  // https://github.com/nerfstudio-project/viser/issues/39
  const removeSceneNode = viewer.useSceneTree((state) => state.removeSceneNode);
  const resetScene = viewer.useSceneTree((state) => state.resetScene);
  const addSceneNode = viewer.useSceneTree((state) => state.addSceneNode);
  const resetGui = viewer.useGui((state) => state.resetGui);
  const setTheme = viewer.useGui((state) => state.setTheme);
  const setShareUrl = viewer.useGui((state) => state.setShareUrl);
  const addGui = viewer.useGui((state) => state.addGui);
  const addModal = viewer.useGui((state) => state.addModal);
  const removeModal = viewer.useGui((state) => state.removeModal);
  const removeGui = viewer.useGui((state) => state.removeGui);
  const updateGuiProps = viewer.useGui((state) => state.updateGuiProps);
  const setClickable = viewer.useSceneTree((state) => state.setClickable);
  const updateUploadState = viewer.useGui((state) => state.updateUploadState);

  // Same as addSceneNode, but make a parent in the form of a dummy coordinate
  // frame if it doesn't exist yet.
  function addSceneNodeMakeParents(node: SceneNode<any>) {
    // Make sure scene node is in attributes.
    const attrs = viewer.nodeAttributesFromName.current;
    attrs[node.name] = {
      overrideVisibility: attrs[node.name]?.overrideVisibility,
    };

    // Don't update the pose of the object until we've made a new one!
    attrs[node.name]!.poseUpdateState = "waitForMakeObject";

    // Make sure parents exists.
    const nodeFromName = viewer.useSceneTree.getState().nodeFromName;
    const parentName = node.name.split("/").slice(0, -1).join("/");
    if (!(parentName in nodeFromName)) {
      addSceneNodeMakeParents(
        new SceneNode<THREE.Group>(parentName, (ref) => (
          <CoordinateFrame ref={ref} showAxes={false} />
        )),
      );
    }
    addSceneNode(node);
  }

  const fileDownloadHandler = useFileDownloadHandler();

  // Return message handler.
  return (message: Message) => {
    if (isGuiConfig(message)) {
      addGui(message);
      return;
    }

    switch (message.type) {
      // Set the share URL.
      case "ShareUrlUpdated": {
        setShareUrl(message.share_url);
        return;
      }
      // Request a render.
      case "GetRenderRequestMessage": {
        viewer.getRenderRequest.current = message;
        viewer.getRenderRequestState.current = "triggered";
        return;
      }
      // Set the GUI panel label.
      case "SetGuiPanelLabelMessage": {
        viewer.useGui.setState({ label: message.label ?? "" });
        return;
      }
      // Configure the theme.
      case "ThemeConfigurationMessage": {
        setTheme(message);
        return;
      }

      // Run some arbitrary Javascript.
      // This is used for plotting, where the Python server will send over a
      // copy of plotly.min.js for the currently-installed version of plotly.
      case "RunJavascriptMessage": {
        eval(message.source);
        return;
      }

      // Add a notification.
      case "NotificationMessage": {
        if (message.mode === "show") {
          notifications.show({
            id: message.id,
            title: message.title,
            message: message.body,
            withCloseButton: message.with_close_button,
            loading: message.loading,
            autoClose: message.auto_close,
            color: message.color ?? undefined,
          });
        } else if (message.mode === "update") {
          notifications.update({
            id: message.id,
            title: message.title,
            message: message.body,
            withCloseButton: message.with_close_button,
            loading: message.loading,
            autoClose: message.auto_close,
            color: message.color ?? undefined,
          });
        }
        return;
      }

      // Remove a specific notification.
      case "RemoveNotificationMessage": {
        notifications.hide(message.id);
        return;
      }
      // Enable/disable whether scene pointer events are sent.
      case "ScenePointerEnableMessage": {
        // Update scene click enable state.
        viewer.scenePointerInfo.current!.enabled = message.enable
          ? message.event_type
          : false;

        // Update cursor to indicate whether the scene can be clicked.
        viewer.canvasRef.current!.style.cursor = message.enable
          ? "pointer"
          : "auto";
        return;
      }

      // Add a coordinate frame.
      case "FrameMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => (
            <CoordinateFrame
              ref={ref}
              showAxes={message.show_axes}
              axesLength={message.axes_length}
              axesRadius={message.axes_radius}
              originRadius={message.origin_radius}
            />
          )),
        );
        return;
      }

      // Add axes to visualize.
      case "BatchedAxesMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => (
              // Minor naming discrepancy: I think "batched" will be clearer to
              // folks on the Python side, but instanced is somewhat more
              // precise.
              <InstancedAxes
                ref={ref}
                wxyzsBatched={
                  new Float32Array(
                    message.wxyzs_batched.buffer.slice(
                      message.wxyzs_batched.byteOffset,
                      message.wxyzs_batched.byteOffset +
                        message.wxyzs_batched.byteLength,
                    ),
                  )
                }
                positionsBatched={
                  new Float32Array(
                    message.positions_batched.buffer.slice(
                      message.positions_batched.byteOffset,
                      message.positions_batched.byteOffset +
                        message.positions_batched.byteLength,
                    ),
                  )
                }
                axes_length={message.axes_length}
                axes_radius={message.axes_radius}
              />
            ),
            undefined,
            undefined,
            undefined,
            // Compute click instance index from instance ID. Each visualized
            // frame has 1 instance for each of 3 line segments.
            (instanceId) => Math.floor(instanceId! / 3),
          ),
        );
        return;
      }

      case "GridMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => (
            <group ref={ref}>
              <Grid
                args={[
                  message.width,
                  message.height,
                  message.width_segments,
                  message.height_segments,
                ]}
                side={THREE.DoubleSide}
                cellColor={message.cell_color}
                cellThickness={message.cell_thickness}
                cellSize={message.cell_size}
                sectionColor={message.section_color}
                sectionThickness={message.section_thickness}
                sectionSize={message.section_size}
                rotation={
                  // There's redundancy here when we set the side to
                  // THREE.DoubleSide, where xy and yx should be the same.
                  //
                  // But it makes sense to keep this parameterization because
                  // specifying planes by xy seems more natural than the normal
                  // direction (z, +z, or -z), and it opens the possibility of
                  // rendering only FrontSide or BackSide grids in the future.
                  //
                  // If we add support for FrontSide or BackSide, we should
                  // double-check that the normal directions from each of these
                  // rotations match the right-hand rule!
                  message.plane == "xz"
                    ? new THREE.Euler(0.0, 0.0, 0.0)
                    : message.plane == "xy"
                    ? new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)
                    : message.plane == "yx"
                    ? new THREE.Euler(0.0, Math.PI / 2.0, Math.PI / 2.0)
                    : message.plane == "yz"
                    ? new THREE.Euler(0.0, 0.0, Math.PI / 2.0)
                    : message.plane == "zx"
                    ? new THREE.Euler(0.0, Math.PI / 2.0, 0.0)
                    : message.plane == "zy"
                    ? new THREE.Euler(-Math.PI / 2.0, 0.0, -Math.PI / 2.0)
                    : undefined
                }
              />
            </group>
          )),
        );
        return;
      }

      // Add a point cloud.
      case "PointCloudMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Points>(message.name, (ref) => (
            <PointCloud
              ref={ref}
              pointSize={message.point_size}
              pointBallNorm={message.point_ball_norm}
              points={
                new Float32Array(
                  message.points.buffer.slice(
                    message.points.byteOffset,
                    message.points.byteOffset + message.points.byteLength,
                  ),
                )
              }
              colors={new Float32Array(message.colors).map(
                (val) => val / 255.0,
              )}
            />
          )),
        );
        return;
      }

      case "GuiModalMessage": {
        addModal(message);
        return;
      }

      case "GuiCloseModalMessage": {
        removeModal(message.id);
        return;
      }

      // Add mesh
      case "SkinnedMeshMessage":
      case "MeshMessage": {
        const geometry = new THREE.BufferGeometry();

        const generateGradientMap = (shades: 3 | 5) => {
          const texture = new THREE.DataTexture(
            Uint8Array.from(
              shades == 3
                ? [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]
                : [
                    0, 0, 0, 255, 64, 64, 64, 255, 128, 128, 128, 255, 192, 192,
                    192, 255, 255, 255, 255, 255,
                  ],
            ),
            shades,
            1,
            THREE.RGBAFormat,
          );

          texture.needsUpdate = true;
          return texture;
        };
        const standardArgs = {
          color: message.color ?? undefined,
          vertexColors: message.vertex_colors !== null,
          wireframe: message.wireframe,
          transparent: message.opacity !== null,
          opacity: message.opacity ?? 1.0,
          // Flat shading only makes sense for non-wireframe materials.
          flatShading: message.flat_shading && !message.wireframe,
          side: {
            front: THREE.FrontSide,
            back: THREE.BackSide,
            double: THREE.DoubleSide,
          }[message.side],
        };
        const assertUnreachable = (x: never): never => {
          throw new Error(`Should never get here! ${x}`);
        };
        const material =
          message.material == "standard" || message.wireframe
            ? new THREE.MeshStandardMaterial(standardArgs)
            : message.material == "toon3"
            ? new THREE.MeshToonMaterial({
                gradientMap: generateGradientMap(3),
                ...standardArgs,
              })
            : message.material == "toon5"
            ? new THREE.MeshToonMaterial({
                gradientMap: generateGradientMap(5),
                ...standardArgs,
              })
            : assertUnreachable(message.material);
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(
              message.vertices.buffer.slice(
                message.vertices.byteOffset,
                message.vertices.byteOffset + message.vertices.byteLength,
              ),
            ),
            3,
          ),
        );
        if (message.vertex_colors !== null) {
          geometry.setAttribute(
            "color",
            threeColorBufferFromUint8Buffer(message.vertex_colors),
          );
        }

        geometry.setIndex(
          new THREE.Uint32BufferAttribute(
            new Uint32Array(
              message.faces.buffer.slice(
                message.faces.byteOffset,
                message.faces.byteOffset + message.faces.byteLength,
              ),
            ),
            1,
          ),
        );
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        const cleanupMesh = () => {
          // TODO: we can switch to the react-three-fiber <bufferGeometry />,
          // <meshStandardMaterial />, etc components to avoid manual
          // disposal.
          geometry.dispose();
          material.dispose();
        };
        if (message.type === "MeshMessage")
          // Normal mesh.
          addSceneNodeMakeParents(
            new SceneNode<THREE.Mesh>(
              message.name,
              (ref) => {
                return (
                  <mesh ref={ref} geometry={geometry} material={material}>
                    <OutlinesIfHovered alwaysMounted />
                  </mesh>
                );
              },
              cleanupMesh,
            ),
          );
        else if (message.type === "SkinnedMeshMessage") {
          // Skinned mesh.
          const bones: THREE.Bone[] = [];
          for (let i = 0; i < message.bone_wxyzs!.length; i++) {
            bones.push(new THREE.Bone());
          }

          const xyzw_quat = new THREE.Quaternion();
          const boneInverses: THREE.Matrix4[] = [];
          viewer.skinnedMeshState.current[message.name] = {
            initialized: false,
            poses: [],
          };
          bones.forEach((bone, i) => {
            const wxyz = message.bone_wxyzs[i];
            const position = message.bone_positions[i];
            xyzw_quat.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);

            const boneInverse = new THREE.Matrix4();
            boneInverse.makeRotationFromQuaternion(xyzw_quat);
            boneInverse.setPosition(position[0], position[1], position[2]);
            boneInverse.invert();
            boneInverses.push(boneInverse);

            bone.quaternion.copy(xyzw_quat);
            bone.position.set(position[0], position[1], position[2]);
            bone.matrixAutoUpdate = false;
            bone.matrixWorldAutoUpdate = false;

            viewer.skinnedMeshState.current[message.name].poses.push({
              wxyz: wxyz,
              position: position,
            });
          });
          const skeleton = new THREE.Skeleton(bones, boneInverses);

          geometry.setAttribute(
            "skinIndex",
            new THREE.Uint16BufferAttribute(
              new Uint16Array(
                message.skin_indices.buffer.slice(
                  message.skin_indices.byteOffset,
                  message.skin_indices.byteOffset +
                    message.skin_indices.byteLength,
                ),
              ),
              4,
            ),
          );
          geometry.setAttribute(
            "skinWeight",
            new THREE.Float32BufferAttribute(
              new Float32Array(
                message.skin_weights!.buffer.slice(
                  message.skin_weights!.byteOffset,
                  message.skin_weights!.byteOffset +
                    message.skin_weights!.byteLength,
                ),
              ),
              4,
            ),
          );

          addSceneNodeMakeParents(
            new SceneNode<THREE.SkinnedMesh>(
              message.name,
              (ref) => {
                return (
                  <skinnedMesh
                    ref={ref}
                    geometry={geometry}
                    material={material}
                    skeleton={skeleton}
                    // TODO: leaving culling on (default) sometimes causes the
                    // mesh to randomly disappear, as of r3f==8.16.2.
                    //
                    // Probably this is because we don't update the bounding
                    // sphere after the bone transforms change.
                    frustumCulled={false}
                  >
                    <OutlinesIfHovered alwaysMounted />
                  </skinnedMesh>
                );
              },
              () => {
                delete viewer.skinnedMeshState.current[message.name];
                skeleton.dispose();
                cleanupMesh();
              },
              false,
              // everyFrameCallback: update bone transforms.
              () => {
                const parentNode = viewer.nodeRefFromName.current[message.name];
                if (parentNode === undefined) return;

                const state = viewer.skinnedMeshState.current[message.name];
                bones.forEach((bone, i) => {
                  if (!state.initialized) {
                    parentNode.add(bone);
                  }
                  const wxyz = state.initialized
                    ? state.poses[i].wxyz
                    : message.bone_wxyzs[i];
                  const position = state.initialized
                    ? state.poses[i].position
                    : message.bone_positions[i];

                  xyzw_quat.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
                  bone.matrix.makeRotationFromQuaternion(xyzw_quat);
                  bone.matrix.setPosition(
                    position[0],
                    position[1],
                    position[2],
                  );
                  bone.updateMatrixWorld();
                });

                if (!state.initialized) {
                  state.initialized = true;
                }
              },
            ),
          );
        }
        return;
      }
      // Set the bone poses.
      case "SetBoneOrientationMessage": {
        const bonePoses = viewer.skinnedMeshState.current;
        bonePoses[message.name].poses[message.bone_index].wxyz = message.wxyz;
        break;
      }
      case "SetBonePositionMessage": {
        const bonePoses = viewer.skinnedMeshState.current;
        bonePoses[message.name].poses[message.bone_index].position =
          message.position;
        break;
      }
      // Add a camera frustum.
      case "CameraFrustumMessage": {
        let texture = undefined;
        if (
          message.image_media_type !== null &&
          message.image_binary !== null
        ) {
          const image_url = URL.createObjectURL(
            new Blob([message.image_binary]),
          );
          texture = new TextureLoader().load(image_url, () =>
            URL.revokeObjectURL(image_url),
          );
        }

        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => (
              <CameraFrustum
                ref={ref}
                fov={message.fov}
                aspect={message.aspect}
                scale={message.scale}
                color={message.color}
                thickness={message.thickness}
                image={texture}
              />
            ),
            () => texture?.dispose(),
          ),
        );
        return;
      }
      case "TransformControlsMessage": {
        const name = message.name;
        const sendDragMessage = makeThrottledMessageSender(viewer, 50);
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => (
              <group onClick={(e) => e.stopPropagation()}>
                <PivotControls
                  ref={ref}
                  scale={message.scale}
                  lineWidth={message.line_width}
                  fixed={message.fixed}
                  autoTransform={message.auto_transform}
                  activeAxes={message.active_axes}
                  disableAxes={message.disable_axes}
                  disableSliders={message.disable_sliders}
                  disableRotations={message.disable_rotations}
                  disableScaling={true}
                  translationLimits={message.translation_limits}
                  rotationLimits={message.rotation_limits}
                  depthTest={message.depth_test}
                  opacity={message.opacity}
                  onDrag={(l) => {
                    const attrs = viewer.nodeAttributesFromName.current;
                    if (attrs[message.name] === undefined) {
                      attrs[message.name] = {};
                    }

                    const wxyz = new THREE.Quaternion();
                    wxyz.setFromRotationMatrix(l);
                    const position = new THREE.Vector3().setFromMatrixPosition(
                      l,
                    );

                    const nodeAttributes = attrs[message.name]!;
                    nodeAttributes.wxyz = [wxyz.w, wxyz.x, wxyz.y, wxyz.z];
                    nodeAttributes.position = position.toArray();
                    sendDragMessage({
                      type: "TransformControlsUpdateMessage",
                      name: name,
                      wxyz: nodeAttributes.wxyz,
                      position: nodeAttributes.position,
                    });
                  }}
                />
              </group>
            ),
            undefined,
            true, // unmountWhenInvisible
          ),
        );
        return;
      }
      case "SetCameraLookAtMessage": {
        const cameraControls = viewer.cameraControlRef.current!;

        const T_threeworld_world = computeT_threeworld_world(viewer);
        const target = new THREE.Vector3(
          message.look_at[0],
          message.look_at[1],
          message.look_at[2],
        );
        target.applyMatrix4(T_threeworld_world);
        cameraControls.setTarget(target.x, target.y, target.z, false);
        return;
      }
      case "SetCameraUpDirectionMessage": {
        const camera = viewer.cameraRef.current!;
        const cameraControls = viewer.cameraControlRef.current!;
        const T_threeworld_world = computeT_threeworld_world(viewer);
        const updir = new THREE.Vector3(
          message.position[0],
          message.position[1],
          message.position[2],
        )
          .normalize()
          .applyQuaternion(
            new THREE.Quaternion().setFromRotationMatrix(T_threeworld_world),
          );
        camera.up.set(updir.x, updir.y, updir.z);

        // Back up position.
        const prevPosition = new THREE.Vector3();
        cameraControls.getPosition(prevPosition);

        cameraControls.updateCameraUp();

        // Restore position, which can get unexpectedly mutated in updateCameraUp().
        cameraControls.setPosition(
          prevPosition.x,
          prevPosition.y,
          prevPosition.z,
          false,
        );
        return;
      }
      case "SetCameraPositionMessage": {
        const cameraControls = viewer.cameraControlRef.current!;

        // Set the camera position. Due to the look-at, note that this will
        // shift the orientation as-well.
        const position_cmd = new THREE.Vector3(
          message.position[0],
          message.position[1],
          message.position[2],
        );

        const T_threeworld_world = computeT_threeworld_world(viewer);
        position_cmd.applyMatrix4(T_threeworld_world);

        cameraControls.setPosition(
          position_cmd.x,
          position_cmd.y,
          position_cmd.z,
        );
        return;
      }
      case "SetCameraFovMessage": {
        const camera = viewer.cameraRef.current!;
        // tan(fov / 2.0) = 0.5 * film height / focal length
        // focal length = 0.5 * film height / tan(fov / 2.0)
        camera.setFocalLength(
          (0.5 * camera.getFilmHeight()) / Math.tan(message.fov / 2.0),
        );
        viewer.sendCameraRef.current !== null && viewer.sendCameraRef.current();
        return;
      }
      case "SetOrientationMessage": {
        const attr = viewer.nodeAttributesFromName.current;
        if (attr[message.name] === undefined) attr[message.name] = {};
        attr[message.name]!.wxyz = message.wxyz;
        if (attr[message.name]!.poseUpdateState == "updated")
          attr[message.name]!.poseUpdateState = "needsUpdate";
        break;
      }
      case "SetPositionMessage": {
        const attr = viewer.nodeAttributesFromName.current;
        if (attr[message.name] === undefined) attr[message.name] = {};
        attr[message.name]!.position = message.position;
        if (attr[message.name]!.poseUpdateState == "updated")
          attr[message.name]!.poseUpdateState = "needsUpdate";
        break;
      }
      case "SetSceneNodeVisibilityMessage": {
        const attr = viewer.nodeAttributesFromName.current;
        if (attr[message.name] === undefined) attr[message.name] = {};
        attr[message.name]!.visibility = message.visible;
        break;
      }
      // Add a background image.
      case "BackgroundImageMessage": {
        const rgb_url = URL.createObjectURL(
          new Blob([message.rgb_bytes], {
            type: message.media_type,
          }),
        );
        new TextureLoader().load(rgb_url, (texture) => {
          URL.revokeObjectURL(rgb_url);
          const oldBackgroundTexture =
            viewer.backgroundMaterialRef.current!.uniforms.colorMap.value;
          viewer.backgroundMaterialRef.current!.uniforms.colorMap.value =
            texture;
          if (isTexture(oldBackgroundTexture)) oldBackgroundTexture.dispose();

          viewer.useGui.setState({ backgroundAvailable: true });
        });
        viewer.backgroundMaterialRef.current!.uniforms.enabled.value = true;
        viewer.backgroundMaterialRef.current!.uniforms.hasDepth.value =
          message.depth_bytes !== null;

        if (message.depth_bytes !== null) {
          // If depth is available set the texture
          const depth_url = URL.createObjectURL(
            new Blob([message.depth_bytes], {
              type: message.media_type,
            }),
          );
          new TextureLoader().load(depth_url, (texture) => {
            URL.revokeObjectURL(depth_url);
            const oldDepthTexture =
              viewer.backgroundMaterialRef.current?.uniforms.depthMap.value;
            viewer.backgroundMaterialRef.current!.uniforms.depthMap.value =
              texture;
            if (isTexture(oldDepthTexture)) oldDepthTexture.dispose();
          });
        }
        return;
      }
      // Add a 2D label.
      case "LabelMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => {
              // We wrap with <group /> because Html doesn't implement THREE.Object3D.
              return (
                <group ref={ref}>
                  <Html>
                    <div
                      style={{
                        width: "10em",
                        fontSize: "0.8em",
                        transform: "translateX(0.1em) translateY(0.5em)",
                      }}
                    >
                      <span
                        style={{
                          background: "#fff",
                          border: "1px solid #777",
                          borderRadius: "0.2em",
                          color: "#333",
                          padding: "0.2em",
                        }}
                      >
                        {message.text}
                      </span>
                    </div>
                  </Html>
                </group>
              );
            },
            undefined,
            true,
          ),
        );
        return;
      }
      case "Gui3DMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => {
              // We wrap with <group /> because Html doesn't implement
              // THREE.Object3D. The initial position is intended to be
              // off-screen; it will be overwritten with the actual position
              // after the component is mounted.
              return (
                <group ref={ref} position={new THREE.Vector3(1e8, 1e8, 1e8)}>
                  <Html>
                    <ContextBridge>
                      <Paper
                        style={{
                          width: "18em",
                          fontSize: "0.875em",
                          marginLeft: "0.5em",
                          marginTop: "0.5em",
                        }}
                        shadow="0 0 0.8em 0 rgba(0,0,0,0.1)"
                        pb="0.25em"
                        onPointerDown={(evt) => {
                          evt.stopPropagation();
                        }}
                      >
                        <ViewerContext.Provider value={viewer}>
                          <GeneratedGuiContainer
                            containerId={message.container_id}
                          />
                        </ViewerContext.Provider>
                      </Paper>
                    </ContextBridge>
                  </Html>
                </group>
              );
            },
            undefined,
            true,
          ),
        );
        return;
      }
      // Add an image.
      case "ImageMessage": {
        // This current implementation may flicker when the image is updated,
        // because the texture is not necessarily done loading before the
        // component is mounted. We could fix this by passing an `onLoad`
        // callback into `TextureLoader`, but this would require work because
        // `addSceneNodeMakeParents` needs to be called immediately: it
        // overwrites position/wxyz attributes, and we don't want this to
        // happen after later messages are received.
        const image_url = URL.createObjectURL(
          new Blob([message.data], {
            type: message.media_type,
          }),
        );
        const texture = new TextureLoader().load(
          image_url,
          () => URL.revokeObjectURL(image_url), // Revoke URL on load.
        );
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(
            message.name,
            (ref) => {
              return (
                <group ref={ref}>
                  <mesh rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}>
                    <OutlinesIfHovered />
                    <planeGeometry
                      attach="geometry"
                      args={[message.render_width, message.render_height]}
                    />
                    <meshBasicMaterial
                      attach="material"
                      transparent={true}
                      side={THREE.DoubleSide}
                      map={texture}
                      toneMapped={false}
                    />
                  </mesh>
                </group>
              );
            },
            () => texture.dispose(),
          ),
        );
        return;
      }
      // Remove a scene node and its children by name.
      case "RemoveSceneNodeMessage": {
        console.log("Removing scene node:", message.name);
        removeSceneNode(message.name);
        const attrs = viewer.nodeAttributesFromName.current;
        delete attrs[message.name];
        return;
      }
      // Set the clickability of a particular scene node.
      case "SetSceneNodeClickableMessage": {
        // This setTimeout is totally unnecessary, but can help surface some race
        // conditions.
        setTimeout(() => setClickable(message.name, message.clickable), 50);
        return;
      }
      // Reset the entire scene, removing all scene nodes.
      case "ResetSceneMessage": {
        resetScene();

        const oldBackground = viewer.sceneRef.current?.background;
        viewer.sceneRef.current!.background = null;
        if (isTexture(oldBackground)) oldBackground.dispose();

        viewer.useGui.setState({ backgroundAvailable: false });
        // Disable the depth texture rendering
        viewer.backgroundMaterialRef.current!.uniforms.enabled.value = false;
        return;
      }
      // Reset the GUI state.
      case "ResetGuiMessage": {
        resetGui();
        return;
      }
      // Update props of a GUI component
      case "GuiUpdateMessage": {
        updateGuiProps(message.id, message.updates);
        return;
      }
      // Remove a GUI input.
      case "GuiRemoveMessage": {
        removeGui(message.id);
        return;
      }
      // Add a glTF/GLB asset.
      case "GlbMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            return (
              <GlbAsset
                ref={ref}
                glb_data={new Uint8Array(message.glb_data)}
                scale={message.scale}
              />
            );
          }),
        );
        return;
      }
      case "CatmullRomSplineMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            return (
              <group ref={ref}>
                <CatmullRomLine
                  points={message.positions}
                  closed={message.closed}
                  curveType={message.curve_type}
                  tension={message.tension}
                  lineWidth={message.line_width}
                  color={message.color}
                  // Sketchy cast needed due to https://github.com/pmndrs/drei/issues/1476.
                  segments={(message.segments ?? undefined) as undefined}
                ></CatmullRomLine>
              </group>
            );
          }),
        );
        return;
      }
      case "CubicBezierSplineMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            return (
              <group ref={ref}>
                {[...Array(message.positions.length - 1).keys()].map((i) => (
                  <CubicBezierLine
                    key={i}
                    start={message.positions[i]}
                    end={message.positions[i + 1]}
                    midA={message.control_points[2 * i]}
                    midB={message.control_points[2 * i + 1]}
                    lineWidth={message.line_width}
                    color={message.color}
                    // Sketchy cast needed due to https://github.com/pmndrs/drei/issues/1476.
                    segments={(message.segments ?? undefined) as undefined}
                  ></CubicBezierLine>
                ))}
              </group>
            );
          }),
        );
        return;
      }
      case "GaussianSplatsMessage": {
        addSceneNodeMakeParents(
          new SceneNode<THREE.Group>(message.name, (ref) => {
            return (
              <SplatObject
                ref={ref}
                buffer={
                  new Uint32Array(
                    message.buffer.buffer.slice(
                      message.buffer.byteOffset,
                      message.buffer.byteOffset + message.buffer.byteLength,
                    ),
                  )
                }
              />
            );
          }),
        );
        return;
      }
      case "FileTransferStart":
      case "FileTransferPart": {
        fileDownloadHandler(message);
        return;
      }
      case "FileTransferPartAck": {
        updateUploadState({
          componentId: message.source_component_id!,
          uploadedBytes: message.transferred_bytes,
          totalBytes: message.total_bytes,
        });
        return;
      }
      default: {
        console.log("Received message did not match any known types:", message);
        return;
      }
    }
  };
}

function useFileDownloadHandler() {
  const downloadStatesRef = React.useRef<{
    [uuid: string]: {
      metadata: FileTransferStart;
      notificationId: string;
      parts: Uint8Array[];
      bytesDownloaded: number;
      displayFilesize: string;
    };
  }>({});

  return (message: FileTransferStart | FileTransferPart) => {
    const notificationId = "download-" + message.transfer_uuid;

    // Create or update download state.
    switch (message.type) {
      case "FileTransferStart": {
        let displaySize = message.size_bytes;
        const displayUnits = ["B", "K", "M", "G", "T", "P"];
        let displayUnitIndex = 0;
        while (
          displaySize >= 100 &&
          displayUnitIndex < displayUnits.length - 1
        ) {
          displaySize /= 1024;
          displayUnitIndex += 1;
        }
        downloadStatesRef.current[message.transfer_uuid] = {
          metadata: message,
          notificationId: notificationId,
          parts: [],
          bytesDownloaded: 0,
          displayFilesize: `${displaySize.toFixed(1)}${
            displayUnits[displayUnitIndex]
          }`,
        };
        break;
      }
      case "FileTransferPart": {
        const downloadState = downloadStatesRef.current[message.transfer_uuid];
        if (message.part != downloadState.parts.length) {
          console.error(
            "A file download message was dropped; this should never happen!",
          );
        }
        downloadState.parts.push(message.content);
        downloadState.bytesDownloaded += message.content.length;
        break;
      }
    }

    // Show notification.
    const downloadState = downloadStatesRef.current[message.transfer_uuid];
    const progressValue =
      (100.0 * downloadState.bytesDownloaded) /
      downloadState.metadata.size_bytes;
    const isDone =
      downloadState.bytesDownloaded == downloadState.metadata.size_bytes;

    (downloadState.bytesDownloaded == 0
      ? notifications.show
      : notifications.update)({
      title:
        (isDone ? "Downloaded " : "Downloading ") +
        `${downloadState.metadata.filename} (${downloadState.displayFilesize})`,
      message: <Progress size="sm" value={progressValue} />,
      id: notificationId,
      autoClose: isDone,
      withCloseButton: isDone,
      loading: !isDone,
      icon: isDone ? <IconCheck /> : undefined,
    });

    // If done: download file and clear state.
    if (isDone) {
      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(
        new Blob(downloadState.parts, {
          type: downloadState.metadata.mime_type,
        }),
      );
      link.download = downloadState.metadata.filename;
      link.click();
      link.remove();
      delete downloadStatesRef.current[message.transfer_uuid];
    }
  };
}

export function FrameSynchronizedMessageHandler() {
  const handleMessage = useMessageHandler();
  const viewer = useContext(ViewerContext)!;
  const messageQueueRef = viewer.messageQueueRef;

  useFrame(() => {
    // Send a render along if it was requested!
    if (viewer.getRenderRequestState.current === "triggered") {
      viewer.getRenderRequestState.current = "pause";
    } else if (viewer.getRenderRequestState.current === "pause") {
      const sourceCanvas = viewer.canvasRef.current!;

      const targetWidth = viewer.getRenderRequest.current!.width;
      const targetHeight = viewer.getRenderRequest.current!.height;

      // We'll save a render to an intermediate canvas with the requested dimensions.
      const renderBufferCanvas = new OffscreenCanvas(targetWidth, targetHeight);
      const ctx = renderBufferCanvas.getContext("2d")!;
      ctx.reset();
      // Use a white background for JPEGs, which don't have an alpha channel.
      if (viewer.getRenderRequest.current?.format === "image/jpeg") {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, renderBufferCanvas.width, renderBufferCanvas.height);
      }

      // Determine offsets for the source canvas. We'll always center our renders.
      // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
      let sourceWidth = sourceCanvas.width;
      let sourceHeight = sourceCanvas.height;

      const sourceAspect = sourceWidth / sourceHeight;
      const targetAspect = targetWidth / targetHeight;

      if (sourceAspect > targetAspect) {
        // The source is wider than the target.
        // We need to shrink the width.
        sourceWidth = Math.round(targetAspect * sourceHeight);
      } else if (sourceAspect < targetAspect) {
        // The source is narrower than the target.
        // We need to shrink the height.
        sourceHeight = Math.round(sourceWidth / targetAspect);
      }

      console.log(
        `Sending render; requested aspect ratio was ${targetAspect} (dimensinos: ${targetWidth}/${targetHeight}), copying from aspect ratio ${
          sourceWidth / sourceHeight
        } (dimensions: ${sourceWidth}/${sourceHeight}).`,
      );

      ctx.drawImage(
        sourceCanvas,
        (sourceCanvas.width - sourceWidth) / 2.0,
        (sourceCanvas.height - sourceHeight) / 2.0,
        sourceWidth,
        sourceHeight,
        0,
        0,
        targetWidth,
        targetHeight,
      );

      viewer.getRenderRequestState.current = "in_progress";

      // Encode the image, the send it.
      renderBufferCanvas
        .convertToBlob({
          type: viewer.getRenderRequest.current!.format,
          quality: viewer.getRenderRequest.current!.quality / 100.0,
        })
        .then(async (blob) => {
          if (blob === null) {
            console.error("Render failed");
            viewer.getRenderRequestState.current = "ready";
            return;
          }
          const payload = new Uint8Array(await blob.arrayBuffer());
          viewer.sendMessageRef.current({
            type: "GetRenderResponseMessage",
            payload: payload,
          });
          viewer.getRenderRequestState.current = "ready";
        });
    }

    // Handle messages, but only if we're not trying to render something.
    if (viewer.getRenderRequestState.current === "ready") {
      // Handle messages before every frame.
      // Place this directly in ws.onmessage can cause race conditions!
      //
      // If a render is requested, note that we don't handle any more messages
      // until the render is done.
      const requestRenderIndex = messageQueueRef.current.findIndex(
        (message) => message.type === "GetRenderRequestMessage",
      );
      const numMessages =
        requestRenderIndex !== -1
          ? requestRenderIndex + 1
          : messageQueueRef.current.length;
      const processBatch = messageQueueRef.current.splice(0, numMessages);
      processBatch.forEach(handleMessage);
    }
  });

  return null;
}
