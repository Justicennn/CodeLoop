"""Compatibility entrypoint forwarding to the Interaction Layer."""

from .interaction.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
