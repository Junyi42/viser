"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import time
import sys
from pathlib import Path

import numpy as onp
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf


def main(
    data_path: Path = Path("/ssd2/junyi/dust3r/checkpoints/eval_sintel_monocular_depth_3datasets_3_7_epoch32_tmp0.01_swinstride5_flow0.01_0.2_35_gt_mask_iter300_fullseq/0/alley_2"),
    downsample_factor: int = 1,
    max_frames: int = 100,
    share: bool = False,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    server.scene.set_up_direction('-z')

    print("Loading frames!")
    loader = viser.extras.Record3dLoader_Customized(data_path)
    num_frames = min(max_frames, loader.num_frames())

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=loader.fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    bg_positions = []
    bg_colors = []
    for i in tqdm(range(num_frames)): # remove last frame since there's no gt mask
        frame = loader.get_frame(i)
        position, color, bg_position, bg_color = frame.get_point_cloud(downsample_factor)

        bg_positions.append(bg_position)
        bg_colors.append(bg_color)

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position,
            colors=color,
            point_size=0.001,
            point_shape="rounded",
        )

        # Place the frustum.
        fov = 2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
        aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.02,
            image=frame.rgb[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # add background frame
    bg_positions = onp.concatenate(bg_positions, axis=0)
    bg_colors = onp.concatenate(bg_colors, axis=0)
    server.scene.add_point_cloud(
        name=f"/frames/background",
        points=bg_positions,
        colors=bg_colors,
        point_size=0.001,
        point_shape="rounded",
    )

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    # get the path to the data with first argument of the command
    try:
        data_path = Path(sys.argv[1])
    except IndexError:
        data_path = Path("/ssd2/junyi/dust3r/checkpoints/eval_sintel_monocular_depth_3datasets_3_7_epoch32_tmp0.01_swinstride5_flow0.01_0.2_35_gt_mask_iter300_fullseq/0/alley_2")
    tyro.cli(main(data_path=data_path))
