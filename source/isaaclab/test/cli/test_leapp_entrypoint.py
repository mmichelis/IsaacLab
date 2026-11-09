# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed LEAPP commands."""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def test_deploy_module_does_not_import_pxr_before_app_launch():
    """Importing deploy must not load pxr before SimulationApp starts."""
    env = os.environ.copy()
    env.update({"ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "Y"})
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import isaaclab.cli.commands.deploy; assert 'pxr' not in sys.modules"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_export_discovers_tasks_after_disabling_jit():
    """The top-level export CLI discovers eager task imports once, with TorchScript disabled."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import sys
                from pathlib import Path
                from types import SimpleNamespace
                from unittest import mock

                import torch
                import isaaclab.cli as cli
                from isaaclab_rl.entrypoints import dispatch

                assert torch.jit._state._enabled
                assert "isaaclab.utils.math" not in sys.modules
                loaded = []

                def load_task():
                    enabled = bool(torch.jit._state._enabled)
                    import isaaclab.utils.math as math_utils
                    loaded.append((enabled, type(math_utils.unscale_transform).__name__))

                entry_point = SimpleNamespace(load=load_task, name="eager-task", dist=SimpleNamespace(name="test"))
                with (
                    mock.patch.object(cli.importlib.metadata, "entry_points", return_value=[entry_point]),
                    mock.patch.object(cli, "DEFAULT_ISAAC_SIM_PATH", Path(sys.executable)),
                    mock.patch.object(dispatch, "_run_backend", return_value=0) as backend,
                    mock.patch.object(sys, "argv", ["isaaclab", "leapp", "export", "--rl_library", "rsl_rl"]),
                ):
                    cli.cli()

                assert loaded == [(False, "function")], loaded
                backend.assert_called_once_with("isaaclab_rl.entrypoints.backends.export_rsl_rl", [])
                """),
        ],
        env={**os.environ, "PYTORCH_JIT": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
