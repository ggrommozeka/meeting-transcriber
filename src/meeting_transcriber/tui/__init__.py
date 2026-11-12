"""Interactive wizards.

The questionary flows are the primary keyboard-first UI. The Textual apps remain
as a hidden fallback via ``MT_TEXTUAL_FALLBACK=1``.
"""

from .app import ProcessApp, SetupApp
from .process_flow import run_process_flow
from .setup_flow import run_setup_flow

__all__ = [
    "ProcessApp",
    "SetupApp",
    "run_process_flow",
    "run_setup_flow",
    "run_process",
    "run_setup",
]


def run_setup() -> None:
    from .app import run_setup as textual_setup

    textual_setup()


def run_process(input_path=None, output_dir=None) -> None:
    from .app import run_process as textual_process

    textual_process(input_path=input_path, output_dir=output_dir)

