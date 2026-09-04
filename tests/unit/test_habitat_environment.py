from __future__ import annotations

import os

from envs.habitat.environment import _quiet_native_output


def test_quiet_native_output_restores_process_streams(capfd) -> None:
    os.write(1, b"before-out\n")
    os.write(2, b"before-err\n")
    with _quiet_native_output(True):
        os.write(1, b"hidden-out\n")
        os.write(2, b"hidden-err\n")
    os.write(1, b"after-out\n")
    os.write(2, b"after-err\n")

    stdout, stderr = capfd.readouterr()
    assert stdout == "before-out\nafter-out\n"
    assert stderr == "before-err\nafter-err\n"
