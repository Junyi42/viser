"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""
# for davis dataset:
# python viser/examples/07_record3d_visualizer_dynamic_dust3r.py --fg_conf_thre 0. --conf_thre 0.001 --point_size 5e-4 --data checkpoints/eval_sintel_4datasets_3_7_epoch20_tmp0.01_swinstride5_flow0.015_0.2_25_gt_mask_iter300_fullseq_davis/0/breakdance-flare
import time
import sys
import argparse
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
    conf_threshold: float = 1.0,
    foreground_conf_threshold: float = 0.1,
    point_size: float = 0.001,
    camera_frustum_scale: float = 0.02,
    no_mask: bool = False,
    xyzw: bool = True,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    server.scene.set_up_direction('-z')

    print("Loading frames!")
    loader = viser.extras.Record3dLoader_Customized(data_path, 
                                                    conf_threshold=conf_threshold, 
                                                    foreground_conf_threshold=foreground_conf_threshold,
                                                    no_mask=no_mask,
                                                    xyzw=xyzw,
                                                    )
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
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)

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
        gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
        gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
        gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        if not gui_show_all_frames.value:
            with server.atomic():
                frame_nodes[current_timestep].visible = True
                frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Show or hide all frames based on the checkbox.
    @gui_show_all_frames.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            # Show all frames
            with server.atomic():
                for frame_node in frame_nodes:
                    frame_node.visible = True
            # Disable playback controls
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True
        else:
            # Show only the current frame
            current_timestep = gui_timestep.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = i == current_timestep
            # Re-enable playback controls
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

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
            point_size=point_size,
            point_shape="rounded",
        )

        # Place the frustum.
        fov = 2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
        aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=camera_frustum_scale,
            image=frame.rgb[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=camera_frustum_scale*2.5,
            axes_radius=camera_frustum_scale/4,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value if not gui_show_all_frames.value else True

    # add background frame
    bg_positions = onp.concatenate(bg_positions, axis=0)
    bg_colors = onp.concatenate(bg_colors, axis=0)
    server.scene.add_point_cloud(
        name=f"/frames/background",
        points=bg_positions,
        colors=bg_colors,
        point_size=point_size,
        point_shape="rounded",
    )

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Process input arguments.")

    # Define arguments
    parser.add_argument(
        "--data", 
        type=Path, 
        nargs="?", 
        default=Path("/ssd2/junyi/dust3r/checkpoints/eval_sintel_monocular_depth_3datasets_3_7_epoch32_tmp0.01_swinstride5_flow0.01_0.2_35_gt_mask_iter300_fullseq/0/alley_2"),
        help="Path to the data"
    )
    parser.add_argument(
        "--conf_thre", 
        type=float, 
        default=0.1, 
        help="Confidence threshold, default is 1.0"
    )
    parser.add_argument(
        "--fg_conf_thre", 
        type=float, 
        default=0.0, 
        help="Foreground confidence threshold, default is 0.1"
    )
    parser.add_argument(
        "--point_size", 
        type=float, 
        default=0.001, 
        help="Point size, default is 0.001"
    )
    parser.add_argument(
        "--camera_size",
        type=float,
        default=0.02,
        help="Camera frustum scale, default is 0.02"
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Don't use mask to filter out points",
    )
    parser.add_argument(
        "--wxyz",
        action="store_true",
        help="Use wxyz for SO3 representation",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    tyro.cli(main(
        data_path=args.data, 
        conf_threshold=args.conf_thre, 
        foreground_conf_threshold=args.fg_conf_thre,
        point_size=args.point_size,
        camera_frustum_scale=args.camera_size,
        no_mask=args.no_mask,
        xyzw=not args.wxyz,
        ))
