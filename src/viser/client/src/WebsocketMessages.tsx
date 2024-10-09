// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.
/** Message for running some arbitrary Javascript on the client.
 * We use this to set up the Plotly.js package, via the plotly.min.js source
 * code.
 *
 * (automatically generated)
 */
export interface RunJavascriptMessage {
  type: "RunJavascriptMessage";
  source: string;
}
/** Notification message.
 *
 * (automatically generated)
 */
export interface NotificationMessage {
  type: "NotificationMessage";
  mode: "show" | "update";
  id: string;
  title: string;
  body: string;
  loading: boolean;
  with_close_button: boolean;
  auto_close: number | false;
  color:
    | "dark"
    | "gray"
    | "red"
    | "pink"
    | "grape"
    | "violet"
    | "indigo"
    | "blue"
    | "cyan"
    | "green"
    | "lime"
    | "yellow"
    | "orange"
    | "teal"
    | null;
}
/** Remove a specific notification.
 *
 * (automatically generated)
 */
export interface RemoveNotificationMessage {
  type: "RemoveNotificationMessage";
  id: string;
}
/** Message for a posed viewer camera.
 * Pose is in the form T_world_camera, OpenCV convention, +Z forward.
 *
 * (automatically generated)
 */
export interface ViewerCameraMessage {
  type: "ViewerCameraMessage";
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
  aspect: number;
  look_at: [number, number, number];
  up_direction: [number, number, number];
}
/** Message for a raycast-like pointer in the scene.
 * origin is the viewing camera position, in world coordinates.
 * direction is the vector if a ray is projected from the camera through the clicked pixel,
 *
 *
 * (automatically generated)
 */
export interface ScenePointerMessage {
  type: "ScenePointerMessage";
  event_type: "click" | "rect-select";
  ray_origin: [number, number, number] | null;
  ray_direction: [number, number, number] | null;
  screen_pos: [number, number][];
}
/** Message to enable/disable scene click events.
 *
 * (automatically generated)
 */
export interface ScenePointerEnableMessage {
  type: "ScenePointerEnableMessage";
  enable: boolean;
  event_type: "click" | "rect-select";
}
/** Variant of CameraMessage used for visualizing camera frustums.
 *
 * OpenCV convention, +Z forward.
 *
 * (automatically generated)
 */
export interface CameraFrustumMessage {
  type: "CameraFrustumMessage";
  name: string;
  fov: number;
  aspect: number;
  scale: number;
  color: number;
  thickness: number;
  image_media_type: "image/jpeg" | "image/png" | null;
  image_binary: Uint8Array | null;
}
/** GlTF message.
 *
 * (automatically generated)
 */
export interface GlbMessage {
  type: "GlbMessage";
  name: string;
  glb_data: Uint8Array;
  scale: number;
}
/** Coordinate frame message.
 *
 * (automatically generated)
 */
export interface FrameMessage {
  type: "FrameMessage";
  name: string;
  show_axes: boolean;
  axes_length: number;
  axes_radius: number;
  origin_radius: number;
}
/** Batched axes message.
 *
 * Positions and orientations should follow a `T_parent_local` convention, which
 * corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`.
 *
 * (automatically generated)
 */
export interface BatchedAxesMessage {
  type: "BatchedAxesMessage";
  name: string;
  wxyzs_batched: Uint8Array;
  positions_batched: Uint8Array;
  axes_length: number;
  axes_radius: number;
}
/** Grid message. Helpful for visualizing things like ground planes.
 *
 * (automatically generated)
 */
export interface GridMessage {
  type: "GridMessage";
  name: string;
  width: number;
  height: number;
  width_segments: number;
  height_segments: number;
  plane: "xz" | "xy" | "yx" | "yz" | "zx" | "zy";
  cell_color: number;
  cell_thickness: number;
  cell_size: number;
  section_color: number;
  section_thickness: number;
  section_size: number;
}
/** Add a 2D label to the scene.
 *
 * (automatically generated)
 */
export interface LabelMessage {
  type: "LabelMessage";
  name: string;
  text: string;
}
/** Add a 3D gui element to the scene.
 *
 * (automatically generated)
 */
export interface Gui3DMessage {
  type: "Gui3DMessage";
  order: number;
  name: string;
  container_id: string;
}
/** Point cloud message.
 *
 * Positions are internally canonicalized to float32, colors to uint8.
 *
 * Float color inputs should be in the range [0,1], int color inputs should be in the
 * range [0,255].
 *
 * (automatically generated)
 */
export interface PointCloudMessage {
  type: "PointCloudMessage";
  name: string;
  points: Uint8Array;
  colors: Uint8Array;
  point_size: number;
  point_ball_norm: number;
}
/** Message for a bone of a skinned mesh.
 *
 * (automatically generated)
 */
export interface MeshBoneMessage {
  type: "MeshBoneMessage";
  name: string;
}
/** Mesh message.
 *
 * Vertices are internally canonicalized to float32, faces to uint32.
 *
 * (automatically generated)
 */
export interface MeshMessage {
  type: "MeshMessage";
  name: string;
  vertices: Uint8Array;
  faces: Uint8Array;
  color: number | null;
  vertex_colors: Uint8Array | null;
  wireframe: boolean;
  opacity: number | null;
  flat_shading: boolean;
  side: "front" | "back" | "double";
  material: "standard" | "toon3" | "toon5";
}
/** Mesh message.
 *
 * Vertices are internally canonicalized to float32, faces to uint32.
 *
 * (automatically generated)
 */
export interface SkinnedMeshMessage {
  type: "SkinnedMeshMessage";
  name: string;
  vertices: Uint8Array;
  faces: Uint8Array;
  color: number | null;
  vertex_colors: Uint8Array | null;
  wireframe: boolean;
  opacity: number | null;
  flat_shading: boolean;
  side: "front" | "back" | "double";
  material: "standard" | "toon3" | "toon5";
  bone_wxyzs: [number, number, number, number][];
  bone_positions: [number, number, number][];
  skin_indices: Uint8Array;
  skin_weights: Uint8Array;
}
/** Server -> client message to set a skinned mesh bone's orientation.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetBoneOrientationMessage {
  type: "SetBoneOrientationMessage";
  name: string;
  bone_index: number;
  wxyz: [number, number, number, number];
}
/** Server -> client message to set a skinned mesh bone's position.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetBonePositionMessage {
  type: "SetBonePositionMessage";
  name: string;
  bone_index: number;
  position: [number, number, number];
}
/** Message for transform gizmos.
 *
 * (automatically generated)
 */
export interface TransformControlsMessage {
  type: "TransformControlsMessage";
  name: string;
  scale: number;
  line_width: number;
  fixed: boolean;
  auto_transform: boolean;
  active_axes: [boolean, boolean, boolean];
  disable_axes: boolean;
  disable_sliders: boolean;
  disable_rotations: boolean;
  translation_limits: [[number, number], [number, number], [number, number]];
  rotation_limits: [[number, number], [number, number], [number, number]];
  depth_test: boolean;
  opacity: number;
}
/** Server -> client message to set the camera's position.
 *
 * (automatically generated)
 */
export interface SetCameraPositionMessage {
  type: "SetCameraPositionMessage";
  position: [number, number, number];
}
/** Server -> client message to set the camera's up direction.
 *
 * (automatically generated)
 */
export interface SetCameraUpDirectionMessage {
  type: "SetCameraUpDirectionMessage";
  position: [number, number, number];
}
/** Server -> client message to set the camera's look-at point.
 *
 * (automatically generated)
 */
export interface SetCameraLookAtMessage {
  type: "SetCameraLookAtMessage";
  look_at: [number, number, number];
}
/** Server -> client message to set the camera's field of view.
 *
 * (automatically generated)
 */
export interface SetCameraFovMessage {
  type: "SetCameraFovMessage";
  fov: number;
}
/** Server -> client message to set a scene node's orientation.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetOrientationMessage {
  type: "SetOrientationMessage";
  name: string;
  wxyz: [number, number, number, number];
}
/** Server -> client message to set a scene node's position.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetPositionMessage {
  type: "SetPositionMessage";
  name: string;
  position: [number, number, number];
}
/** Client -> server message when a transform control is updated.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface TransformControlsUpdateMessage {
  type: "TransformControlsUpdateMessage";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
}
/** Message for rendering a background image.
 *
 * (automatically generated)
 */
export interface BackgroundImageMessage {
  type: "BackgroundImageMessage";
  media_type: "image/jpeg" | "image/png";
  rgb_bytes: Uint8Array;
  depth_bytes: Uint8Array | null;
}
/** Message for rendering 2D images.
 *
 * (automatically generated)
 */
export interface ImageMessage {
  type: "ImageMessage";
  name: string;
  media_type: "image/jpeg" | "image/png";
  data: Uint8Array;
  render_width: number;
  render_height: number;
}
/** Remove a particular node from the scene.
 *
 * (automatically generated)
 */
export interface RemoveSceneNodeMessage {
  type: "RemoveSceneNodeMessage";
  name: string;
}
/** Set the visibility of a particular node in the scene.
 *
 * (automatically generated)
 */
export interface SetSceneNodeVisibilityMessage {
  type: "SetSceneNodeVisibilityMessage";
  name: string;
  visible: boolean;
}
/** Set the clickability of a particular node in the scene.
 *
 * (automatically generated)
 */
export interface SetSceneNodeClickableMessage {
  type: "SetSceneNodeClickableMessage";
  name: string;
  clickable: boolean;
}
/** Message for clicked objects.
 *
 * (automatically generated)
 */
export interface SceneNodeClickMessage {
  type: "SceneNodeClickMessage";
  name: string;
  instance_index: number | null;
  ray_origin: [number, number, number];
  ray_direction: [number, number, number];
  screen_pos: [number, number];
}
/** Reset scene.
 *
 * (automatically generated)
 */
export interface ResetSceneMessage {
  type: "ResetSceneMessage";
}
/** Reset GUI.
 *
 * (automatically generated)
 */
export interface ResetGuiMessage {
  type: "ResetGuiMessage";
}
/** GuiAddFolderMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', expand_by_default: 'bool', visible: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddFolderMessage {
  type: "GuiAddFolderMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  expand_by_default: boolean;
  visible: boolean;
}
/** GuiAddMarkdownMessage(order: 'float', id: 'str', markdown: 'str', container_id: 'str', visible: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddMarkdownMessage {
  type: "GuiAddMarkdownMessage";
  order: number;
  id: string;
  markdown: string;
  container_id: string;
  visible: boolean;
}
/** GuiAddProgressBarMessage(order: 'float', id: 'str', value: 'float', animated: 'bool', color: 'Optional[Color]', container_id: 'str', visible: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddProgressBarMessage {
  type: "GuiAddProgressBarMessage";
  order: number;
  id: string;
  value: number;
  animated: boolean;
  color:
    | "dark"
    | "gray"
    | "red"
    | "pink"
    | "grape"
    | "violet"
    | "indigo"
    | "blue"
    | "cyan"
    | "green"
    | "lime"
    | "yellow"
    | "orange"
    | "teal"
    | null;
  container_id: string;
  visible: boolean;
}
/** GuiAddPlotlyMessage(order: 'float', id: 'str', plotly_json_str: 'str', aspect: 'float', container_id: 'str', visible: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddPlotlyMessage {
  type: "GuiAddPlotlyMessage";
  order: number;
  id: string;
  plotly_json_str: string;
  aspect: number;
  container_id: string;
  visible: boolean;
}
/** GuiAddTabGroupMessage(order: 'float', id: 'str', container_id: 'str', tab_labels: 'Tuple[str, ...]', tab_icons_html: 'Tuple[Union[str, None], ...]', tab_container_ids: 'Tuple[str, ...]', visible: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddTabGroupMessage {
  type: "GuiAddTabGroupMessage";
  order: number;
  id: string;
  container_id: string;
  tab_labels: string[];
  tab_icons_html: (string | null)[];
  tab_container_ids: string[];
  visible: boolean;
}
/** Base message type containing fields commonly used by GUI inputs.
 *
 * (automatically generated)
 */
export interface _GuiAddInputBase {
  type: "_GuiAddInputBase";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: any;
  visible: boolean;
  disabled: boolean;
}
/** GuiAddButtonMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'bool', visible: 'bool', disabled: 'bool', color: 'Optional[Color]', icon_html: 'Optional[str]')
 *
 * (automatically generated)
 */
export interface GuiAddButtonMessage {
  type: "GuiAddButtonMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: boolean;
  visible: boolean;
  disabled: boolean;
  color:
    | "dark"
    | "gray"
    | "red"
    | "pink"
    | "grape"
    | "violet"
    | "indigo"
    | "blue"
    | "cyan"
    | "green"
    | "lime"
    | "yellow"
    | "orange"
    | "teal"
    | null;
  icon_html: string | null;
}
/** GuiAddUploadButtonMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'Any', visible: 'bool', disabled: 'bool', color: 'Optional[Color]', icon_html: 'Optional[str]', mime_type: 'str')
 *
 * (automatically generated)
 */
export interface GuiAddUploadButtonMessage {
  type: "GuiAddUploadButtonMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: any;
  visible: boolean;
  disabled: boolean;
  color:
    | "dark"
    | "gray"
    | "red"
    | "pink"
    | "grape"
    | "violet"
    | "indigo"
    | "blue"
    | "cyan"
    | "green"
    | "lime"
    | "yellow"
    | "orange"
    | "teal"
    | null;
  icon_html: string | null;
  mime_type: string;
}
/** GuiAddSliderMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'float', visible: 'bool', disabled: 'bool', min: 'float', max: 'float', step: 'Optional[float]', precision: 'int', marks: 'Optional[Tuple[GuiSliderMark, ...]]' = None)
 *
 * (automatically generated)
 */
export interface GuiAddSliderMessage {
  type: "GuiAddSliderMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: number;
  visible: boolean;
  disabled: boolean;
  min: number;
  max: number;
  step: number | null;
  precision: number;
  marks: { value: number; label?: string }[] | null;
}
/** GuiAddMultiSliderMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'Any', visible: 'bool', disabled: 'bool', min: 'float', max: 'float', step: 'Optional[float]', min_range: 'Optional[float]', precision: 'int', fixed_endpoints: 'bool' = False, marks: 'Optional[Tuple[GuiSliderMark, ...]]' = None)
 *
 * (automatically generated)
 */
export interface GuiAddMultiSliderMessage {
  type: "GuiAddMultiSliderMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: any;
  visible: boolean;
  disabled: boolean;
  min: number;
  max: number;
  step: number | null;
  min_range: number | null;
  precision: number;
  fixed_endpoints: boolean;
  marks: { value: number; label?: string }[] | null;
}
/** GuiAddNumberMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'float', visible: 'bool', disabled: 'bool', precision: 'int', step: 'float', min: 'Optional[float]', max: 'Optional[float]')
 *
 * (automatically generated)
 */
export interface GuiAddNumberMessage {
  type: "GuiAddNumberMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: number;
  visible: boolean;
  disabled: boolean;
  precision: number;
  step: number;
  min: number | null;
  max: number | null;
}
/** GuiAddRgbMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'Tuple[int, int, int]', visible: 'bool', disabled: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddRgbMessage {
  type: "GuiAddRgbMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: [number, number, number];
  visible: boolean;
  disabled: boolean;
}
/** GuiAddRgbaMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'Tuple[int, int, int, int]', visible: 'bool', disabled: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddRgbaMessage {
  type: "GuiAddRgbaMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: [number, number, number, number];
  visible: boolean;
  disabled: boolean;
}
/** GuiAddCheckboxMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'bool', visible: 'bool', disabled: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddCheckboxMessage {
  type: "GuiAddCheckboxMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: boolean;
  visible: boolean;
  disabled: boolean;
}
/** GuiAddVector2Message(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'Tuple[float, float]', visible: 'bool', disabled: 'bool', min: 'Optional[Tuple[float, float]]', max: 'Optional[Tuple[float, float]]', step: 'float', precision: 'int')
 *
 * (automatically generated)
 */
export interface GuiAddVector2Message {
  type: "GuiAddVector2Message";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: [number, number];
  visible: boolean;
  disabled: boolean;
  min: [number, number] | null;
  max: [number, number] | null;
  step: number;
  precision: number;
}
/** GuiAddVector3Message(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'Tuple[float, float, float]', visible: 'bool', disabled: 'bool', min: 'Optional[Tuple[float, float, float]]', max: 'Optional[Tuple[float, float, float]]', step: 'float', precision: 'int')
 *
 * (automatically generated)
 */
export interface GuiAddVector3Message {
  type: "GuiAddVector3Message";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: [number, number, number];
  visible: boolean;
  disabled: boolean;
  min: [number, number, number] | null;
  max: [number, number, number] | null;
  step: number;
  precision: number;
}
/** GuiAddTextMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'str', visible: 'bool', disabled: 'bool')
 *
 * (automatically generated)
 */
export interface GuiAddTextMessage {
  type: "GuiAddTextMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: string;
  visible: boolean;
  disabled: boolean;
}
/** GuiAddDropdownMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'str', visible: 'bool', disabled: 'bool', options: 'Tuple[str, ...]')
 *
 * (automatically generated)
 */
export interface GuiAddDropdownMessage {
  type: "GuiAddDropdownMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: string;
  visible: boolean;
  disabled: boolean;
  options: string[];
}
/** GuiAddButtonGroupMessage(order: 'float', id: 'str', label: 'str', container_id: 'str', hint: 'Optional[str]', value: 'str', visible: 'bool', disabled: 'bool', options: 'Tuple[str, ...]')
 *
 * (automatically generated)
 */
export interface GuiAddButtonGroupMessage {
  type: "GuiAddButtonGroupMessage";
  order: number;
  id: string;
  label: string;
  container_id: string;
  hint: string | null;
  value: string;
  visible: boolean;
  disabled: boolean;
  options: string[];
}
/** GuiModalMessage(order: 'float', id: 'str', title: 'str')
 *
 * (automatically generated)
 */
export interface GuiModalMessage {
  type: "GuiModalMessage";
  order: number;
  id: string;
  title: string;
}
/** GuiCloseModalMessage(id: 'str')
 *
 * (automatically generated)
 */
export interface GuiCloseModalMessage {
  type: "GuiCloseModalMessage";
  id: string;
}
/** Sent server->client to remove a GUI element.
 *
 * (automatically generated)
 */
export interface GuiRemoveMessage {
  type: "GuiRemoveMessage";
  id: string;
}
/** Sent client<->server when any property of a GUI component is changed.
 *
 * (automatically generated)
 */
export interface GuiUpdateMessage {
  type: "GuiUpdateMessage";
  id: string;
  updates: Partial<GuiAddComponentMessage>;
}
/** Message from server->client to configure parts of the GUI.
 *
 * (automatically generated)
 */
export interface ThemeConfigurationMessage {
  type: "ThemeConfigurationMessage";
  titlebar_content: {
    buttons:
      | {
          text: string | null;
          icon: "GitHub" | "Description" | "Keyboard" | null;
          href: string | null;
        }[]
      | null;
    image: {
      image_url_light: string;
      image_url_dark: string | null;
      image_alt: string;
      href: string | null;
    } | null;
  } | null;
  control_layout: "floating" | "collapsible" | "fixed";
  control_width: "small" | "medium" | "large";
  show_logo: boolean;
  show_share_button: boolean;
  dark_mode: boolean;
  colors:
    | [
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
      ]
    | null;
}
/** Message from server->client carrying Catmull-Rom spline information.
 *
 * (automatically generated)
 */
export interface CatmullRomSplineMessage {
  type: "CatmullRomSplineMessage";
  name: string;
  positions: [number, number, number][];
  curve_type: "centripetal" | "chordal" | "catmullrom";
  tension: number;
  closed: boolean;
  line_width: number;
  color: number;
  segments: number | null;
}
/** Message from server->client carrying Cubic Bezier spline information.
 *
 * (automatically generated)
 */
export interface CubicBezierSplineMessage {
  type: "CubicBezierSplineMessage";
  name: string;
  positions: [number, number, number][];
  control_points: [number, number, number][];
  line_width: number;
  color: number;
  segments: number | null;
}
/** Message from server->client carrying splattable Gaussians.
 *
 * (automatically generated)
 */
export interface GaussianSplatsMessage {
  type: "GaussianSplatsMessage";
  name: string;
  buffer: Uint8Array;
}
/** Message from server->client requesting a render of the current viewport.
 *
 * (automatically generated)
 */
export interface GetRenderRequestMessage {
  type: "GetRenderRequestMessage";
  format: "image/jpeg" | "image/png";
  height: number;
  width: number;
  quality: number;
}
/** Message from client->server carrying a render.
 *
 * (automatically generated)
 */
export interface GetRenderResponseMessage {
  type: "GetRenderResponseMessage";
  payload: Uint8Array;
}
/** Signal that a file is about to be sent.
 *
 * (automatically generated)
 */
export interface FileTransferStart {
  type: "FileTransferStart";
  source_component_id: string | null;
  transfer_uuid: string;
  filename: string;
  mime_type: string;
  part_count: number;
  size_bytes: number;
}
/** Send a file for clients to download or upload files from client.
 *
 * (automatically generated)
 */
export interface FileTransferPart {
  type: "FileTransferPart";
  source_component_id: string | null;
  transfer_uuid: string;
  part: number;
  content: Uint8Array;
}
/** Send a file for clients to download or upload files from client.
 *
 * (automatically generated)
 */
export interface FileTransferPartAck {
  type: "FileTransferPartAck";
  source_component_id: string | null;
  transfer_uuid: string;
  transferred_bytes: number;
  total_bytes: number;
}
/** Message from client->server to connect to the share URL server.
 *
 * (automatically generated)
 */
export interface ShareUrlRequest {
  type: "ShareUrlRequest";
}
/** Message from server->client to indicate that the share URL has been updated.
 *
 * (automatically generated)
 */
export interface ShareUrlUpdated {
  type: "ShareUrlUpdated";
  share_url: string | null;
}
/** Message from client->server to disconnect from the share URL server.
 *
 * (automatically generated)
 */
export interface ShareUrlDisconnect {
  type: "ShareUrlDisconnect";
}
/** Message from server->client to set the label of the GUI panel.
 *
 * (automatically generated)
 */
export interface SetGuiPanelLabelMessage {
  type: "SetGuiPanelLabelMessage";
  label: string | null;
}

export type Message =
  | RunJavascriptMessage
  | NotificationMessage
  | RemoveNotificationMessage
  | ViewerCameraMessage
  | ScenePointerMessage
  | ScenePointerEnableMessage
  | CameraFrustumMessage
  | GlbMessage
  | FrameMessage
  | BatchedAxesMessage
  | GridMessage
  | LabelMessage
  | Gui3DMessage
  | PointCloudMessage
  | MeshBoneMessage
  | MeshMessage
  | SkinnedMeshMessage
  | SetBoneOrientationMessage
  | SetBonePositionMessage
  | TransformControlsMessage
  | SetCameraPositionMessage
  | SetCameraUpDirectionMessage
  | SetCameraLookAtMessage
  | SetCameraFovMessage
  | SetOrientationMessage
  | SetPositionMessage
  | TransformControlsUpdateMessage
  | BackgroundImageMessage
  | ImageMessage
  | RemoveSceneNodeMessage
  | SetSceneNodeVisibilityMessage
  | SetSceneNodeClickableMessage
  | SceneNodeClickMessage
  | ResetSceneMessage
  | ResetGuiMessage
  | GuiAddFolderMessage
  | GuiAddMarkdownMessage
  | GuiAddProgressBarMessage
  | GuiAddPlotlyMessage
  | GuiAddTabGroupMessage
  | _GuiAddInputBase
  | GuiAddButtonMessage
  | GuiAddUploadButtonMessage
  | GuiAddSliderMessage
  | GuiAddMultiSliderMessage
  | GuiAddNumberMessage
  | GuiAddRgbMessage
  | GuiAddRgbaMessage
  | GuiAddCheckboxMessage
  | GuiAddVector2Message
  | GuiAddVector3Message
  | GuiAddTextMessage
  | GuiAddDropdownMessage
  | GuiAddButtonGroupMessage
  | GuiModalMessage
  | GuiCloseModalMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | ThemeConfigurationMessage
  | CatmullRomSplineMessage
  | CubicBezierSplineMessage
  | GaussianSplatsMessage
  | GetRenderRequestMessage
  | GetRenderResponseMessage
  | FileTransferStart
  | FileTransferPart
  | FileTransferPartAck
  | ShareUrlRequest
  | ShareUrlUpdated
  | ShareUrlDisconnect
  | SetGuiPanelLabelMessage;
export type GuiAddComponentMessage =
  | GuiAddFolderMessage
  | GuiAddMarkdownMessage
  | GuiAddProgressBarMessage
  | GuiAddPlotlyMessage
  | GuiAddTabGroupMessage
  | GuiAddButtonMessage
  | GuiAddUploadButtonMessage
  | GuiAddSliderMessage
  | GuiAddMultiSliderMessage
  | GuiAddNumberMessage
  | GuiAddRgbMessage
  | GuiAddRgbaMessage
  | GuiAddCheckboxMessage
  | GuiAddVector2Message
  | GuiAddVector3Message
  | GuiAddTextMessage
  | GuiAddDropdownMessage
  | GuiAddButtonGroupMessage;
