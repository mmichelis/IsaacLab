# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural dump of a robot USD: joints, mount transforms, collision/mimic counts.

Runs on plain usd-core (no kit). Used for asset-to-asset comparisons; see
SIM2REAL_FINDINGS.md.
"""

import sys

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(sys.argv[1])
print("== defaultPrim:", stage.GetDefaultPrim().GetPath())


def fmt_vec(v):
    return "(" + ", ".join(f"{x:.4f}" for x in v) + ")" if v is not None else "None"


TYPES = (
    ("revolute", "PhysicsRevoluteJoint", UsdPhysics.RevoluteJoint),
    ("prismatic", "PhysicsPrismaticJoint", UsdPhysics.PrismaticJoint),
    ("fixed", "PhysicsFixedJoint", UsdPhysics.FixedJoint),
)

joints = []
for prim in stage.Traverse():
    for label, type_name, cls in TYPES:
        if prim.GetTypeName() != type_name:
            continue
        j = cls(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        row = {
            "name": prim.GetName(),
            "type": label,
            "b0": b0[0].name if b0 else "?",
            "b1": b1[0].name if b1 else "?",
            "pos0": j.GetLocalPos0Attr().Get(),
            "pos1": j.GetLocalPos1Attr().Get(),
            "rot0": j.GetLocalRot0Attr().Get(),
        }
        if label == "revolute":
            row["axis"] = cls(prim).GetAxisAttr().Get()
            row["lo"] = cls(prim).GetLowerLimitAttr().Get()
            row["hi"] = cls(prim).GetUpperLimitAttr().Get()
        joints.append(row)
        break

print(f"== joints ({len(joints)}):")
for r in joints:
    lim = ""
    if r["type"] == "revolute" and r.get("lo") is not None:
        lim = f" axis={r.get('axis')} lim=[{r['lo']:.1f},{r['hi']:.1f}]"
    print(
        f"  {r['type']:9s} {r['name']:38s} {r['b0']:>24s} -> {r['b1']:24s} "
        f"pos0={fmt_vec(r['pos0'])} pos1={fmt_vec(r['pos1'])} rot0={r['rot0']}{lim}"
    )

num_collision = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
num_mimic = 0
for p in stage.Traverse():
    if any("mimic" in name.lower() for name in p.GetPropertyNames()):
        num_mimic += 1
print(f"== collision prims: {num_collision}, prims with mimic props: {num_mimic}")
