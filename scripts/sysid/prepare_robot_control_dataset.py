# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enrich dex/robot-control chirp collections to the sysid data contract.

The robot-control collector (FrankaDriver: libfranka torque mode with a 1 kHz
host PD consuming SHM position targets) predates parts of the fail-closed
contract in ``data_contract.py``: its runs carry a
plain-string ``gains_provenance``, no ``shaper_type``, no ``kp_used/kd_used``
and no ``safety_controller`` dict. All of these are recoverable ground truth:

- the driver applies SHM latest-value position targets with a host PD and no
  command shaping, so ``shaper_type`` is ``none`` by construction;
- the exact rig gains are stamped in ``shm_pd_params_yaml``;
- run completion is stamped as a top-level ``operator_stop`` bool.

This script reads each run's ``franka_fr3_chirp_data.npz`` and writes a
contract-complete ``chirp_data_prepared.pt`` next to the requested output root,
mirroring the input tree. Source files are never modified.

Usage:
    python scripts/sysid/prepare_robot_control_dataset.py \\
        --input_root /datasets/chirp_data_robot_control \\
        --output_root logs/sysid/prepared_datasets
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

# Keys copied through unchanged (tensors converted to torch, scalars to python).
_PASSTHROUGH_TENSORS = ("time", "des_dof_pos", "dof_pos", "dof_vel", "dof_tau_est", "state_stamps", "state_fresh")
_PASSTHROUGH_SCALARS = ("sample_rate", "intended_duration_s", "mode", "robot_name")


def _parse_pd_gains(yaml_text: str) -> tuple[list[float], list[float]]:
    """Extract the active (uncommented) stiffness/damping gain lists from the stamped yaml."""
    gains = {}
    for key in ("stiffness_gains", "damping_gains"):
        # Match only uncommented occurrences; the yaml keeps alternates commented out.
        matches = re.findall(rf"^\s*{key}:\s*\[([^\]]+)\]", yaml_text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one uncommented '{key}' in shm_pd_params_yaml, found {len(matches)}")
        gains[key] = [float(x) for x in matches[0].split(",")]
    return gains["stiffness_gains"], gains["damping_gains"]


def prepare_run(npz_path: Path, out_path: Path) -> None:
    """Convert one robot-control npz collection into a contract-complete .pt dataset."""
    raw = np.load(npz_path, allow_pickle=True)
    data: dict = {}

    for key in _PASSTHROUGH_TENSORS:
        data[key] = torch.as_tensor(np.asarray(raw[key]))
    for key in _PASSTHROUGH_SCALARS:
        value = raw[key]
        data[key] = str(value) if value.dtype.kind == "U" else value.item()
    data["joint_names"] = [str(n) for n in raw["joint_names"]]
    data["active_joint_names"] = [str(n) for n in raw["active_joint_names"]]

    # Rig gains from the stamped PD yaml — the exact values the driver ran with.
    yaml_text = str(raw["shm_pd_params_yaml"])
    kp, kd = _parse_pd_gains(yaml_text)
    num_joints = len(data["joint_names"])
    if len(kp) != num_joints or len(kd) != num_joints:
        raise ValueError(f"gain lists have {len(kp)}/{len(kd)} entries for {num_joints} joints")
    data["kp_used"] = torch.tensor(kp, dtype=torch.float32)
    data["kd_used"] = torch.tensor(kd, dtype=torch.float32)

    # FrankaDriver consumes SHM latest-value targets with no command shaping.
    data["shaper_type"] = "none"
    # The FrankaDriver collector has no clamp/abort channels; operator_stop is
    # the only truncation signal and is stamped top-level.
    data["safety_controller"] = {
        "clamped": False,
        "aborted": False,
        "operator_stop": bool(raw["operator_stop"]),
    }
    # Keep the original pointer, but as the JSON object the contract expects.
    data["gains_provenance"] = json.dumps(
        {
            "command_shaping": "none",
            "source": str(raw["gains_provenance"]),
            "shm_pd_params_yaml": yaml_text,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"[OK] {npz_path} -> {out_path}  (kp={kp}, kd={kd})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_root", type=str, required=True, help="Root of the robot-control collection tree.")
    parser.add_argument("--output_root", type=str, required=True, help="Writable root for the prepared .pt files.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    npz_files = sorted(input_root.rglob("franka_fr3_chirp_data.npz"))
    if not npz_files:
        raise SystemExit(f"no franka_fr3_chirp_data.npz under {input_root}")
    for npz_path in npz_files:
        rel = npz_path.parent.relative_to(input_root)
        prepare_run(npz_path, output_root / rel / "chirp_data_prepared.pt")
    print(f"[DONE] prepared {len(npz_files)} runs under {output_root}")


if __name__ == "__main__":
    main()
