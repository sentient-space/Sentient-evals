from __future__ import annotations

import os
from pathlib import Path

import pytest

from sentient_evals.environments.base import EnvironmentConfig
from sentient_evals.environments.e2b import E2BEnvironment


pytestmark = pytest.mark.asyncio


def _integration_enabled() -> bool:
    return os.environ.get("RUN_E2B_INTEGRATION") == "1" and bool(os.environ.get("E2B_API_KEY"))


@pytest.mark.skipif(not _integration_enabled(), reason="Set RUN_E2B_INTEGRATION=1 and E2B_API_KEY")
async def test_e2b_environment_smoke(tmp_path: Path):
    pytest.importorskip("e2b")

    env = E2BEnvironment(
        trial_id="e2b-smoke",
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
