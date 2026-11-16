"""bottrade — paper-trading bot for Polymarket + pump.fun.

A small, dependency-light multi-agent trading stack:

    feed -> scanner -> sentiment -> whale -> signal -> risk -> executor -> portfolio

Nothing in this package places a real order. The `PaperBroker` simulates fills
against live (or synthetic) quotes so strategies can be validated safely first.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
