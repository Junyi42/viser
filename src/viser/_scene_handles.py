from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Generic, Literal, TypeVar

import numpy as onp

from . import _messages
from .infra._infra import WebsockClientConnection, WebsockServer

if TYPE_CHECKING:
    from ._gui_api import GuiApi
    from ._gui_handles import SupportsRemoveProtocol
    from ._scene_api import SceneApi
    from ._viser import ClientHandle
    from .infra import ClientId


@dataclasses.dataclass(frozen=True)
class ScenePointerEvent:
    """Event passed to pointer callbacks for the scene (currently only clicks)."""

    client: ClientHandle
    """Client that triggered this event."""
    client_id: int
    """ID of client that triggered this event."""
    event_type: _messages.ScenePointerEventType
    """Type of event that was triggered. Currently we only support clicks and box selections."""
    ray_origin: tuple[float, float, float] | None
    """Origin of 3D ray corresponding to this click, in world coordinates."""
    ray_direction: tuple[float, float, float] | None
    """Direction of 3D ray corresponding to this click, in world coordinates."""
    screen_pos: tuple[tuple[float, float], ...]
    """Screen position of the click on the screen (OpenCV image coordinates, 0 to 1).
    (0, 0) is the upper-left corner, (1, 1) is the bottom-right corner.
    For a box selection, this includes the min- and max- corners of the box."""

    @property
    def event(self):
        """Deprecated. Use `event_type` instead."""
        return self.event_type


TSceneNodeHandle = TypeVar("TSceneNodeHandle", bound="SceneNodeHandle")


@dataclasses.dataclass
class _SceneNodeHandleState:
    name: str
    api: SceneApi
    wxyz: onp.ndarray = dataclasses.field(
        default_factory=lambda: onp.array([1.0, 0.0, 0.0, 0.0])
    )
    position: onp.ndarray = dataclasses.field(
        default_factory=lambda: onp.array([0.0, 0.0, 0.0])
    )
    visible: bool = True
    # TODO: we should remove SceneNodeHandle as an argument here.
    click_cb: list[Callable[[SceneNodePointerEvent[SceneNodeHandle]], None]] | None = (
        None
    )


@dataclasses.dataclass
class SceneNodeHandle:
    """Handle base class for interacting with scene nodes."""

    _impl: _SceneNodeHandleState

    @classmethod
    def _make(
        cls: type[TSceneNodeHandle],
        api: SceneApi,
        name: str,
        wxyz: tuple[float, float, float, float] | onp.ndarray,
        position: tuple[float, float, float] | onp.ndarray,
        visible: bool,
    ) -> TSceneNodeHandle:
        out = cls(_SceneNodeHandleState(name, api))
        api._handle_from_node_name[name] = out

        out.wxyz = wxyz
        out.position = position

        # Toggle visibility to make sure we send a
        # SetSceneNodeVisibilityMessage to the client.
        out._impl.visible = not visible
        out.visible = visible
        return out

    @property
    def wxyz(self) -> onp.ndarray:
        """Orientation of the scene node. This is the quaternion representation of the R
        in `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.wxyz

    @wxyz.setter
    def wxyz(self, wxyz: tuple[float, float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        wxyz_cast = cast_vector(wxyz, 4)
        self._impl.wxyz = onp.asarray(wxyz)
        self._impl.api._websock_interface.queue_message(
            _messages.SetOrientationMessage(self._impl.name, wxyz_cast)
        )

    @property
    def position(self) -> onp.ndarray:
        """Position of the scene node. This is equivalent to the t in
        `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.position

    @position.setter
    def position(self, position: tuple[float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        position_cast = cast_vector(position, 3)
        self._impl.position = onp.asarray(position)
        self._impl.api._websock_interface.queue_message(
            _messages.SetPositionMessage(self._impl.name, position_cast)
        )

    @property
    def visible(self) -> bool:
        """Whether the scene node is visible or not. Synchronized to clients automatically when assigned."""
        return self._impl.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible == self._impl.visible:
            return
        self._impl.api._websock_interface.queue_message(
            _messages.SetSceneNodeVisibilityMessage(self._impl.name, visible)
        )
        self._impl.visible = visible

    def remove(self) -> None:
        """Remove the node from the scene."""
        self._impl.api._websock_interface.queue_message(
            _messages.RemoveSceneNodeMessage(self._impl.name)
        )


@dataclasses.dataclass(frozen=True)
class SceneNodePointerEvent(Generic[TSceneNodeHandle]):
    """Event passed to pointer callbacks for scene nodes (currently only clicks)."""

    client: ClientHandle
    """Client that triggered this event."""
    client_id: int
    """ID of client that triggered this event."""
    event: Literal["click"]
    """Type of event that was triggered. Currently we only support clicks."""
    target: TSceneNodeHandle
    """Scene node that was clicked."""
    ray_origin: tuple[float, float, float]
    """Origin of 3D ray corresponding to this click, in world coordinates."""
    ray_direction: tuple[float, float, float]
    """Direction of 3D ray corresponding to this click, in world coordinates."""
    screen_pos: tuple[float, float]
    """Screen position of the click on the screen (OpenCV image coordinates, 0 to 1).
    (0, 0) is the upper-left corner, (1, 1) is the bottom-right corner."""
    instance_index: int | None
    """Instance ID of the clicked object, if applicable. Currently this is `None` for all objects except for the output of :meth:`SceneApi.add_batched_axes()`."""


@dataclasses.dataclass
class _ClickableSceneNodeHandle(SceneNodeHandle):
    def on_click(
        self: TSceneNodeHandle,
        func: Callable[[SceneNodePointerEvent[TSceneNodeHandle]], None],
    ) -> Callable[[SceneNodePointerEvent[TSceneNodeHandle]], None]:
        """Attach a callback for when a scene node is clicked."""
        self._impl.api._websock_interface.queue_message(
            _messages.SetSceneNodeClickableMessage(self._impl.name, True)
        )
        if self._impl.click_cb is None:
            self._impl.click_cb = []
        self._impl.click_cb.append(func)  # type: ignore
        return func


@dataclasses.dataclass
class CameraFrustumHandle(_ClickableSceneNodeHandle):
    """Handle for camera frustums."""


@dataclasses.dataclass
class PointCloudHandle(SceneNodeHandle):
    """Handle for point clouds. Does not support click events."""


@dataclasses.dataclass
class BatchedAxesHandle(_ClickableSceneNodeHandle):
    """Handle for batched coordinate frames."""


@dataclasses.dataclass
class FrameHandle(_ClickableSceneNodeHandle):
    """Handle for coordinate frames."""


@dataclasses.dataclass
class MeshHandle(_ClickableSceneNodeHandle):
    """Handle for mesh objects."""


@dataclasses.dataclass
class GaussianSplatHandle(_ClickableSceneNodeHandle):
    """Handle for Gaussian splatting objects.

    **Work-in-progress.** Gaussian rendering is still under development.
    """


@dataclasses.dataclass
class MeshSkinnedHandle(_ClickableSceneNodeHandle):
    """Handle for skinned mesh objects."""

    bones: tuple[MeshSkinnedBoneHandle, ...]
    """Bones of the skinned mesh. These handles can be used for reading and
    writing poses, which are defined relative to the mesh root."""


@dataclasses.dataclass
class BoneState:
    name: str
    websock_interface: WebsockServer | WebsockClientConnection
    bone_index: int
    wxyz: onp.ndarray
    position: onp.ndarray


@dataclasses.dataclass
class MeshSkinnedBoneHandle:
    """Handle for reading and writing the poses of bones in a skinned mesh."""

    _impl: BoneState

    @property
    def wxyz(self) -> onp.ndarray:
        """Orientation of the bone. This is the quaternion representation of the R
        in `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.wxyz

    @wxyz.setter
    def wxyz(self, wxyz: tuple[float, float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        wxyz_cast = cast_vector(wxyz, 4)
        self._impl.wxyz = onp.asarray(wxyz)
        self._impl.websock_interface.queue_message(
            _messages.SetBoneOrientationMessage(
                self._impl.name, self._impl.bone_index, wxyz_cast
            )
        )

    @property
    def position(self) -> onp.ndarray:
        """Position of the bone. This is equivalent to the t in
        `p_parent = [R | t] p_local`. Synchronized to clients automatically when assigned.
        """
        return self._impl.position

    @position.setter
    def position(self, position: tuple[float, float, float] | onp.ndarray) -> None:
        from ._scene_api import cast_vector

        position_cast = cast_vector(position, 3)
        self._impl.position = onp.asarray(position)
        self._impl.websock_interface.queue_message(
            _messages.SetBonePositionMessage(
                self._impl.name, self._impl.bone_index, position_cast
            )
        )


@dataclasses.dataclass
class GlbHandle(_ClickableSceneNodeHandle):
    """Handle for GLB objects."""


@dataclasses.dataclass
class ImageHandle(_ClickableSceneNodeHandle):
    """Handle for 2D images, rendered in 3D."""


@dataclasses.dataclass
class LabelHandle(SceneNodeHandle):
    """Handle for 2D label objects. Does not support click events."""


@dataclasses.dataclass
class _TransformControlsState:
    last_updated: float
    update_cb: list[Callable[[TransformControlsHandle], None]]
    sync_cb: None | Callable[[ClientId, TransformControlsHandle], None] = None


@dataclasses.dataclass
class TransformControlsHandle(_ClickableSceneNodeHandle):
    """Handle for interacting with transform control gizmos."""

    _impl_aux: _TransformControlsState

    @property
    def update_timestamp(self) -> float:
        return self._impl_aux.last_updated

    def on_update(
        self, func: Callable[[TransformControlsHandle], None]
    ) -> Callable[[TransformControlsHandle], None]:
        """Attach a callback for when the gizmo is moved."""
        self._impl_aux.update_cb.append(func)
        return func


@dataclasses.dataclass
class Gui3dContainerHandle(SceneNodeHandle):
    """Use as a context to place GUI elements into a 3D GUI container."""

    _gui_api: GuiApi
    _container_id: str
    _container_id_restore: str | None = None
    _children: dict[str, SupportsRemoveProtocol] = dataclasses.field(
        default_factory=dict
    )

    def __enter__(self) -> Gui3dContainerHandle:
        self._container_id_restore = self._gui_api._get_container_id()
        self._gui_api._set_container_id(self._container_id)
        return self

    def __exit__(self, *args) -> None:
        del args
        assert self._container_id_restore is not None
        self._gui_api._set_container_id(self._container_id_restore)
        self._container_id_restore = None

    def __post_init__(self) -> None:
        self._gui_api._container_handle_from_id[self._container_id] = self

    def remove(self) -> None:
        """Permanently remove this GUI container from the visualizer."""

        # Call scene node remove.
        super().remove()

        # Clean up contained GUI elements.
        for child in tuple(self._children.values()):
            child.remove()
        self._gui_api._container_handle_from_id.pop(self._container_id)
