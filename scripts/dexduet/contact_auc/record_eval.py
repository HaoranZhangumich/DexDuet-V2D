# SPDX-FileCopyrightText: Copyright (c) 2026 the DexDuet authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only recorder around the unchanged V2D kit evaluator, used for Contact-AUC (reproduce/contact_auc.sh).

The official runtime and the kit come from the environment variables FC_OFFICIAL and V2D_KIT. Invoked from $FC_OFFICIAL:
  cd $FC_OFFICIAL && env PYTHONPATH=$FC_OFFICIAL/src PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    CUDA_VISIBLE_DEVICES=<g> FC_OFFICIAL=... V2D_KIT=... $PY -B record_eval.py \
      --checkpoint <file> --sha256 <hex> --episode <e> --out <rundir>

1. sys.argv = the kit script + the official overrides (kit examples/run_policy_evaluation.py l.71-80).
2. The hook replaces the module attribute flash_chord.evaluation.harness.record_object_pose_rollout with a
   wrapper that (a) wraps jax_env.step on that instance so that, after each call, world 0 of env.state_0.body_q
   (all bodies) is copied on the device (wp.copy, explicit device arrays, no new kernel) into a preallocated
   (T*B) transform buffer, (b) calls the original function with unchanged arguments and returns its result
   unchanged, (c) saves <rundir>/recording.npz. The wrapper on jax_env.step is removed after the rollout.
3. runpy.run_path(kit script, run_name="__main__").
Nothing in the kit, the stock runtime, the config, the seeds or the computation is changed.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import runpy  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ST = Path(os.environ["FC_OFFICIAL"])  # NVIDIA's official FlashCHORD checkout (video_to_data @ 8c778f9)
KIT = Path(os.environ["V2D_KIT"])  # the extracted V2D submission kit
KIT_SCRIPT = KIT / "flash_chord/scripts/evaluate_policy.py"
PINNED = {
    KIT_SCRIPT: "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1",
    ST / "src/flash_chord/evaluation/harness.py": "6e61b092e9aa9d00d11c557b8508c09ef990314a3f68b0779bbf97765c85ddc2",
    ST / "src/flash_chord/evaluation/rollout.py": "7794b59dd52c43fcbffeb5c8f09a91feb35bae11082fda4cf4d237423b1fdeb4",
    ST / "src/flash_chord/evaluation/metrics.py": "ef641481aaefd2792cb106df43ede28b0b52223b30f610be4d28286cc4d0209e",
    ST / "src/flash_chord/envs/base.py": "5011b2c672911d4d0b68fe0f832e6a333943627e5020308fd3bd241619dd3292",
    ST / "src/flash_chord/embodiments/g1_dex3.py": "c61742fb4d9f95c53c4e87cc2c4daa97812cf01db98ba11805d2592d048bffca",
    ST / "src/flash_chord/runtime/command.py": "2d9256c4757f9528afaf5a457047358fcf42d18ad826cadd36aaa5417c72c441",
}
SIDES = ("left", "right")
DIGITS = ("thumb", "middle", "index")  # stock g1_dex3._DIGITS order
TIP_LINK = {"thumb": "hand_thumb_2_link", "middle": "hand_middle_1_link", "index": "hand_index_1_link"}
LINKS = (
    "hand_palm_link",
    "hand_thumb_0_link",
    "hand_thumb_1_link",
    "hand_thumb_2_link",
    "hand_middle_0_link",
    "hand_middle_1_link",
    "hand_index_0_link",
    "hand_index_1_link",
)


def tip_offset(side, digit):
    if digit == "thumb":
        return (0.0, -0.035 if side == "left" else 0.035, 0.0)
    return (0.035, 0.0, 0.0)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rotate_xyzw(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) in xyzw order (float64)."""
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def keypoint_spec(env):
    """Dex3 keypoints per side: 3 stock fingertip frames + the hand links with object_contact geometry."""
    model = env.scene.model
    world_count = int(env.world_count)
    bodies = int(model.body_count) // world_count
    labels = [str(label) for label in model.body_label]
    terminal = [label.rsplit("/", 1)[-1] for label in labels[:bodies]]
    shape_body = np.asarray(model.shape_body.numpy())
    problems = []
    spec = {}
    for hand in env.scene.layout.hands:
        side = hand.side
        names, body_ids, offsets, quats, kinds, expect = [], [], [], [], [], []
        for digit, frame in zip(DIGITS, hand.fingertip_frames):
            names.append(frame.name)
            body_ids.append(int(frame.body_id))
            offsets.append(tuple(float(x) for x in frame.body_to_frame_pos))
            quats.append(tuple(float(x) for x in frame.body_to_frame_quat_xyzw))
            kinds.append("tip")
            expect.append(f"{side}_{TIP_LINK[digit]}")
            if frame.name != f"{side}_{digit}_fingertip":
                problems.append(f"{side}: fingertip frame {frame.name!r} != {side}_{digit}_fingertip")
            if not np.allclose(frame.body_to_frame_pos, tip_offset(side, digit), rtol=0, atol=1e-12):
                problems.append(f"{side}: fingertip {frame.name} offset {frame.body_to_frame_pos} != stock")
            if not np.allclose(frame.body_to_frame_quat_xyzw, (0, 0, 0, 1), rtol=0, atol=1e-12):
                problems.append(f"{side}: fingertip {frame.name} has a rotated frame")
        shape_check = {}
        for geometry in hand.link_geometry:
            if not geometry.object_contact:
                continue
            matches = [b for b in range(bodies) if terminal[b] == geometry.link_name]
            if len(matches) != 1:
                problems.append(f"{side}: link {geometry.link_name} resolves to {matches}")
                continue
            body = matches[0]
            names.append(geometry.link_name)
            body_ids.append(body)
            offsets.append((0.0, 0.0, 0.0))
            quats.append((0.0, 0.0, 0.0, 1.0))
            kinds.append("link")
            expect.append(geometry.link_name)
            owners = sorted({int(shape_body[s]) for s in geometry.shape_ids if 0 <= s < shape_body.size})
            shape_check[geometry.link_name] = owners
        if len(names) != 11:
            problems.append(f"{side}: {len(names)} keypoints, expected 11")
        if tuple(n for n, k in zip(names, kinds) if k == "link") != tuple(f"{side}_{x}" for x in LINKS):
            problems.append(f"{side}: link order {names[3:]} differs from the stock HandLayout order")
        resolved = [terminal[b] if 0 <= b < bodies else None for b in body_ids]
        for name, got, want in zip(names, resolved, expect):
            if got != want:
                problems.append(f"{side}: keypoint {name} body label {got!r} != {want!r}")
        palm_label = terminal[hand.palm_frame.body_id]
        if palm_label != f"{side}_hand_palm_link":
            problems.append(f"{side}: palm frame body label {palm_label!r}")
        spec[side] = {
            "names": names,
            "body_ids": body_ids,
            "offsets": offsets,
            "quats_xyzw": quats,
            "kinds": kinds,
            "resolved_labels": resolved,
            "expected_labels": expect,
            "link_shape_owner_bodies": shape_check,
        }
    sides = [hand.side for hand in env.scene.layout.hands]
    if tuple(sides) != SIDES:
        problems.append(f"hand sides {sides}")
    return spec, labels[:bodies], problems


def object_mesh(scene):
    """Visible object mesh shapes, scale and shape transform applied (as stock sample_object_vertices reads them)."""
    import newton

    from flash_chord.utils.quat import quat_rotate_xyzw

    model = scene.model
    visible = int(newton.ShapeFlags.VISIBLE)
    mesh_type = int(newton.GeoType.MESH)
    flags = np.asarray(model.shape_flags.numpy())
    shape_body = np.asarray(model.shape_body.numpy())
    shape_type = np.asarray(model.shape_type.numpy())
    shape_scale = np.asarray(model.shape_scale.numpy(), dtype=np.float64)
    shape_transform = np.asarray(model.shape_transform.numpy(), dtype=np.float64)
    vertices, faces, pieces = [], [], []
    count = 0
    for binding in scene.objects:
        for body in binding.bodies:
            for shape in range(binding.shapes.start, binding.shapes.stop):
                if shape_body[shape] != body.body_id or shape_type[shape] != mesh_type:
                    continue
                if not flags[shape] & visible:
                    continue
                source = model.shape_source[shape]
                if source is None or not hasattr(source, "vertices"):
                    continue
                scaled = np.asarray(source.vertices, dtype=np.float64) * shape_scale[shape][None, :]
                offset = shape_transform[shape]
                v = quat_rotate_xyzw(offset[3:7], scaled) + offset[:3][None, :]
                f = np.asarray(source.indices, dtype=np.int64).reshape(-1, 3) + count
                vertices.append(v)
                faces.append(f)
                pieces.append(
                    {
                        "object": str(binding.name),
                        "body_id": int(body.body_id),
                        "reference_body_id": int(body.reference_body_id),
                        "shape": int(shape),
                        "vertices": int(v.shape[0]),
                        "faces": int(f.shape[0]),
                        "scale": [float(x) for x in shape_scale[shape]],
                        "transform": [float(x) for x in offset],
                    }
                )
                count += v.shape[0]
    return np.concatenate(vertices), np.concatenate(faces), pieces


class Recorder:
    def __init__(self, rundir, meta):
        self.rundir = rundir
        self.meta = meta
        self.calls = 0
        self.saved = False
        self.problems = []
        self.info = {}

    def install(self):
        import flash_chord.evaluation.harness as harness

        if Path(harness.__file__).resolve() != (ST / "src/flash_chord/evaluation/harness.py").resolve():
            raise SystemExit(f"harness imported from {harness.__file__}, not the stock runtime")
        self.original = harness.record_object_pose_rollout
        if self.original.__module__ != "flash_chord.evaluation.rollout":
            raise SystemExit(f"unexpected record_object_pose_rollout {self.original!r}")
        recorder = self

        def record_object_pose_rollout(*args, **kwargs):
            return recorder.hooked(*args, **kwargs)

        harness.record_object_pose_rollout = record_object_pose_rollout
        if harness.record_object_pose_rollout is not record_object_pose_rollout:
            raise SystemExit("hook could not be installed")

    def hooked(self, *args, **kwargs):
        import warp as wp

        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("record_object_pose_rollout called more than once")
        env, jax_env = args[0], args[1]
        step_count = int(kwargs["step_count"])
        start_frame = int(kwargs.get("start_frame", 0))
        world_count = int(env.world_count)
        if world_count != 1:
            raise RuntimeError(f"recorder expects world_count 1, got {world_count}")
        model = env.scene.model
        bodies = int(model.body_count) // world_count
        device = env.device
        body_q0 = env.state_0.body_q
        buffer = wp.zeros(step_count * bodies, dtype=body_q0.dtype, device=device)
        counter = [0]
        original_step = jax_env.step
        if "step" in vars(jax_env):
            raise RuntimeError("jax_env.step is already an instance attribute")

        def step(action):
            result = original_step(action)
            index = counter[0]
            if index >= step_count:
                raise RuntimeError("more jax_env.step calls than step_count")
            source = env.state_0.body_q
            if source.device != buffer.device:
                raise RuntimeError(f"body_q on {source.device}, buffer on {buffer.device}")
            wp.copy(buffer, source, dest_offset=index * bodies, src_offset=0, count=bodies)
            counter[0] = index + 1
            return result

        jax_env.step = step
        t0 = time.time()
        try:
            rollout = self.original(*args, **kwargs)
        finally:
            del jax_env.step
        t1 = time.time()
        if counter[0] != step_count:
            raise RuntimeError(f"recorded {counter[0]} steps, expected {step_count}")
        body_q = buffer.numpy().reshape(step_count, bodies, 7)
        try:
            self.save(env, rollout, body_q, step_count, start_frame, bodies, t1 - t0)
        except Exception:  # never alter the evaluation; report the recording failure
            self.problems.append("save failed: " + traceback.format_exc())
        return rollout

    def save(self, env, rollout, body_q, step_count, start_frame, bodies, seconds):
        scene = env.scene
        object_ids = np.asarray(env.command.body_ids_w.numpy(), dtype=np.int64)
        layout = env.command.layout
        n_obj_bodies = int(layout.num_bodies)
        achieved = np.asarray(rollout.achieved_pose_w)
        recorded = body_q[:, object_ids[:n_obj_bodies]].astype(np.float64)
        non_finite = tuple(int(w) for w in rollout.non_finite_worlds)
        if non_finite:
            r1a = (not bool(np.isfinite(recorded).all())) and bool(np.all(achieved[:, 0] == 0.0))
        else:
            r1a = bool(np.array_equal(recorded, achieved[:, 0], equal_nan=True))
        spec, labels, kp_problems = keypoint_spec(env)
        self.problems.extend(kp_problems)
        kpts = {}
        for side in SIDES:
            ids = np.asarray(spec[side]["body_ids"], dtype=np.int64)
            offs = np.asarray(spec[side]["offsets"], dtype=np.float64)
            q = body_q[:, ids].astype(np.float64)
            kpts[side] = rotate_xyzw(q[..., 3:7], np.broadcast_to(offs, q[..., :3].shape)) + q[..., :3]
        mesh_v, mesh_f, pieces = object_mesh(scene)
        scales = np.asarray(scene.object_scales, dtype=np.float64)
        try:
            physics_fps = float(env.config.sim.physics_fps)
        except Exception:
            physics_fps = float("nan")
        try:
            dev = wp_device_info(env.device)
        except Exception:
            dev = {}
        arrays = {
            "body_q": body_q.astype(np.float32),
            "body_labels": np.asarray(labels),
            "object_body_ids": object_ids[:n_obj_bodies],
            "object_body_names": np.asarray(list(rollout.object_body_names)),
            "body_object_ids": np.asarray(rollout.body_object_ids, dtype=np.int64),
            "achieved_pose_w": achieved.astype(np.float64),
            "reference_pose_w": np.asarray(rollout.reference_pose_w, dtype=np.float64),
            "object_vertices_o": np.asarray(rollout.object_vertices_o, dtype=np.float64),
            "control_fps": np.float64(rollout.control_fps),
            "physics_fps": np.float64(physics_fps),
            "start_frame": np.int64(rollout.start_frame),
            "step_count": np.int64(step_count),
            "non_finite_worlds": np.asarray(non_finite, dtype=np.int64),
            "object_scales": scales,
            "mesh_vertices_o": mesh_v,
            "mesh_faces": mesh_f,
            "kpts_left": kpts["left"],
            "kpts_right": kpts["right"],
        }
        for side in SIDES:
            arrays[f"kp_{side}_names"] = np.asarray(spec[side]["names"])
            arrays[f"kp_{side}_body_ids"] = np.asarray(spec[side]["body_ids"], dtype=np.int64)
            arrays[f"kp_{side}_offsets"] = np.asarray(spec[side]["offsets"], dtype=np.float64)
            arrays[f"kp_{side}_quats_xyzw"] = np.asarray(spec[side]["quats_xyzw"], dtype=np.float64)
        arrays["meta_json"] = np.asarray(json.dumps(self.meta, sort_keys=True))
        final = self.rundir / "recording.npz"
        temporary = self.rundir / f".recording.{os.getpid()}.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, final)
        self.saved = True
        self.info = {
            "recording": str(final),
            "recording_sha256": sha256(final),
            "rollout_seconds": seconds,
            "step_count": step_count,
            "bodies_per_world": bodies,
            "object_body_ids": [int(x) for x in object_ids[:n_obj_bodies]],
            "object_body_labels": [labels[int(x)] for x in object_ids[:n_obj_bodies]],
            "object_body_names": list(rollout.object_body_names),
            "object_scales": scales.tolist(),
            "mesh_pieces": pieces,
            "mesh_vertex_count": int(mesh_v.shape[0]),
            "mesh_face_count": int(mesh_f.shape[0]),
            "non_finite_worlds": list(non_finite),
            "R1a_body_q_equals_achieved": r1a,
            "keypoints": spec,
            "P7_problems": kp_problems,
            "device": dev,
            "control_fps": float(rollout.control_fps),
            "physics_fps": physics_fps,
            "start_frame": int(rollout.start_frame),
        }


def wp_device_info(device):
    import warp as wp

    d = wp.get_device(device)
    return {"alias": str(d.alias), "name": str(d.name), "uuid": str(getattr(d, "uuid", ""))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rundir = args.out.resolve()
    rundir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:06d}"
    metrics_output = rundir / "results" / f"{stem}.json"
    trajectory_output = rundir / "results" / f"{stem}.parquet"
    if metrics_output.exists() or trajectory_output.exists() or (rundir / "recording.npz").exists():
        raise SystemExit(f"{rundir} already holds outputs; a completed run is never re-run")
    record = {
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": sha256(Path(__file__).resolve()),
        "started_at": now(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "python": sys.executable,
        "episode": args.episode,
        "checkpoint": str(args.checkpoint),
        "expected_sha256": args.sha256,
        "status": "started",
        "problems": [],
    }
    record_path = rundir / "record.json"

    def write_record():
        tmp = rundir / f".record.{os.getpid()}.json"
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
        os.replace(tmp, record_path)

    pinned = {str(path): sha256(path) for path in PINNED}
    record["pinned"] = pinned
    bad = [path for path, digest in PINNED.items() if pinned[str(path)] != digest]
    if bad:
        record["status"] = "pinned_file_changed"
        record["problems"].append(f"pinned files changed: {[str(p) for p in bad]}")
        write_record()
        raise SystemExit(3)
    actual = sha256(args.checkpoint)
    record["checkpoint_sha256"] = actual
    if actual != args.sha256:
        record["status"] = "checkpoint_sha_mismatch"
        write_record()
        raise SystemExit(4)
    if os.environ.get("PYTHONPATH") != str(ST / "src"):
        record["status"] = "bad_pythonpath"
        write_record()
        raise SystemExit(5)
    argv = [
        str(KIT_SCRIPT),
        f"evaluation.checkpoint={args.checkpoint}",
        "evaluation.world_count=1",
        "evaluation.reset_mode=explicit",
        f"evaluation.metrics_output={metrics_output}",
        f"evaluation.object_trajectories_output={trajectory_output}",
        f"hydra.run.dir={rundir / 'results' / 'hydra' / stem}",
        "logging.mode=disabled",
    ]
    record["argv"] = argv
    write_record()
    recorder = Recorder(
        rundir,
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": actual,
            "episode": args.episode,
            "argv": argv,
            "cuda_visible_devices": record["cuda_visible_devices"],
            "started_at": record["started_at"],
            "tool_sha256": record["tool_sha256"],
        },
    )
    recorder.install()
    import flash_chord

    record["flash_chord_module"] = str(Path(flash_chord.__file__).resolve())
    if not Path(flash_chord.__file__).resolve().is_relative_to((ST / "src").resolve()):
        record["status"] = "flash_chord_not_stock"
        write_record()
        raise SystemExit(6)
    sys.argv = argv
    exit_code = 0
    t0 = time.time()
    try:
        runpy.run_path(str(KIT_SCRIPT), run_name="__main__")
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    except BaseException:
        record["problems"].append("kit script raised: " + traceback.format_exc())
        exit_code = 1
    record["kit_seconds"] = time.time() - t0
    record["kit_exit"] = exit_code
    record["finished_at"] = now()
    record["hook_calls"] = recorder.calls
    record["recording_saved"] = recorder.saved
    record["problems"].extend(recorder.problems)
    record.update(recorder.info)
    record["metrics_output_exists"] = metrics_output.is_file()
    record["trajectory_output_exists"] = trajectory_output.is_file()
    ok = exit_code == 0 and recorder.saved and metrics_output.is_file() and trajectory_output.is_file()
    record["status"] = "ok" if ok else "failed"
    write_record()
    print(json.dumps({"status": record["status"], "R1a": record.get("R1a_body_q_equals_achieved")}), flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
