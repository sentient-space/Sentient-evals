from __future__ import annotations

import os
from pathlib import Path

import pytest

from sentient_evals.environments.base import EnvironmentConfig
from sentient_evals.environments.modal import ModalEnvironment


pytestmark = pytest.mark.asyncio


def _integration_enabled() -> bool:
    return (
        os.environ.get("RUN_MODAL_INTEGRATION") == "1"
        and bool(os.environ.get("MODAL_TOKEN_ID"))
        and bool(os.environ.get("MODAL_TOKEN_SECRET"))
    )


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="Set RUN_MODAL_INTEGRATION=1 with MODAL_TOKEN_ID and MODAL_TOKEN_SECRET",
)
async def test_modal_environment_smoke(tmp_path: Path):
    pytest.importorskip("modal")

    env = ModalEnvironment(
        trial_id="modal-smoke",
        workspace_dir=tmp_path / "workspace",
        logs_dir=tmp_path / "logs",
        config=EnvironmentConfig(allow_internet=True, build_timeout_sec=120),
        environment_dir=None,
        image=None,
    )

    await env.start()
    try:
        result = await env.exec("echo smoke-test")
        assert result.exit_code == 0
        assert "smoke-test" in result.stdout
    finally:
        await env.stop(delete=True)
