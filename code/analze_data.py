from pathlib import Path
import matplotlib.pyplot as plt
import mps_motion
import mps


def windowed_mps_data(
    mps_data: mps.MPS,
    analysis_window_start: float = 0,
    analysis_window_end: float = -1,
    startX: int = 0,
    endX: int = -1,
    startY: int = 0,
    endY: int = -1,
) -> mps_motion.utils.MPSData:
    frames = mps_data.frames
    time_stamps = mps_data.time_stamps

    Nx, Ny, Nt = frames.shape
    if endX == -1:
        endX = Nx
    if endY == -1:
        endY = Ny
    start_index = 0
    end_index = Nt
    if analysis_window_start > 0:
        try:
            start_index = next(
                i for i, v in enumerate(time_stamps) if v >= analysis_window_start
            )
        except StopIteration:
            pass
    if analysis_window_end != -1:
        try:
            end_index = next(
                i for i, v in enumerate(time_stamps) if v >= analysis_window_end
            )
        except StopIteration:
            pass

    if startX >= frames.shape[0]:
        raise ValueError(
            f"Cannot create a empty frame sequence. startX = {startX}, but {frames.shape =}",
        )
    if startY >= frames.shape[1]:
        raise ValueError(
            f"Cannot create a empty frame sequence. starty = {startY}, but {frames.shape =}",
        )
    return mps_motion.utils.MPSData(
        frames=frames[startX:endX, startY:endY, start_index:end_index],
        time_stamps=time_stamps[start_index:end_index],
        pacing=mps_data.pacing[start_index:end_index],
        info=mps_data.info,
        metadata=mps_data.metadata,
    )


def analze_motion(data, postfix, outdir, name):
    outdir.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    ax.imshow(data.frames[:, :, 0].T, cmap="gray")
    fig.savefig(outdir / f"{name}_first_frame{postfix}.png")

    mps.utils.frames2mp4(
        data.frames.T,
        outdir / f"{name}_movie{postfix}.mp4",
        framerate=data.framerate,
    )

    opt_flow = mps_motion.OpticalFlow(data, flow_algorithm="farneback")
    v = opt_flow.get_velocities(spacing=1) * 1000.0
    v_norm = v.norm().mean().compute()
    v_x = v.x.mean().compute()
    v_y = v.y.mean().compute()
    v_max = v.norm().max().compute()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    ax[0].plot(data.time_stamps[:-1], v_x)
    ax[0].set_title("$v_x$")
    ax[1].plot(data.time_stamps[:-1], v_y)
    ax[1].set_title("$v_y$")
    ax[2].plot(data.time_stamps[:-1], v_norm)
    ax[2].set_title("$\| v \|$")
    fig.savefig(outdir / f"{name}_velocity{postfix}.png")

    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(v_max.T)
    cbar = fig.colorbar(im)
    cbar.set_label("$\| v \|$")
    fig.savefig(outdir / f"{name}_velocity_max{postfix}.png")

    mps_motion.visu.quiver_video(
        data,
        v,
        outdir / f"{name}_velocity{postfix}.mp4",
        step=24,
        vector_scale=0.1,
        offset=1,
    )

    reference_frame_index = (
        mps_motion.motion_tracking.estimate_referece_image_from_velocity(
            t=data.time_stamps[:-1],
            v=v_norm,
        )
    )
    reference_frame = data.time_stamps[reference_frame_index]

    fig, ax = plt.subplots()
    ax.plot(data.time_stamps[:-1], v_norm)
    ax.plot([reference_frame], [v_norm[reference_frame_index]], "ro")
    fig.savefig(outdir / f"{name}_reference_frame{postfix}.png")

    u = opt_flow.get_displacements(
        reference_frame=reference_frame, smooth_ref_transition=False
    )

    u_norm = u.norm().mean().compute()
    u_x = u.x.mean().compute()
    u_y = u.y.mean().compute()
    u_max = u.norm().max().compute()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    ax[0].plot(data.time_stamps, u_x)
    ax[0].set_title("$u_x$")
    ax[1].plot(data.time_stamps, u_y)
    ax[1].set_title("$u_y$")
    ax[2].plot(data.time_stamps, u_norm)
    ax[2].set_title("$\| u \|$")
    fig.savefig(outdir / f"{name}_displacement{postfix}.png")

    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(u_max.T)
    cbar = fig.colorbar(im)
    cbar.set_label("$\| u \|$")
    fig.savefig(outdir / f"{name}_displacement_max{postfix}.png")

    mps_motion.visu.quiver_video(
        data, u, outdir / f"{name}_displacement{postfix}.mp4", step=24, vector_scale=1
    )


def main(outdir="motion_results"):
    data_folder = Path(
        f"/Users/finsberg/Dropbox/ComPhy/cardiac-cancer/eht-mechanical-forces_2023-04-18_0813"
    )
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    name = "B1"
    data = mps.MPS(data_folder / f"{name}.avi")
    analze_motion(data, "", outdir / name, name)
    selection = windowed_mps_data(data, startX=280, endX=560, startY=750, endY=1500)
    analze_motion(selection, "_selection", outdir / name, name)

    name = "B5"
    data = mps.MPS(data_folder / f"{name}.avi")
    analze_motion(data, "", outdir / name, name)
    selection = windowed_mps_data(data, startX=350, endX=700, startY=750, endY=1500)
    analze_motion(selection, "_selection", outdir / name, name)

    name = "C5"
    data = mps.MPS(data_folder / f"{name}.avi")
    analze_motion(data, "", outdir / name, name)
    selection = windowed_mps_data(data, startX=280, endX=560, startY=750, endY=1500)
    analze_motion(selection, "_selection", outdir / name, name)


if __name__ == "__main__":
    main()
