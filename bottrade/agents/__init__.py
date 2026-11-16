"""Agent pipeline package.

Built-in agents, in pipeline order:

    scanner    build the tradable universe and publish quotes
    sentiment  lexicon / news / LLM probability nudge
    whale      aggressive-flow imbalance from public trade data
    signal     combine nudges into a fair price and emit signals
    risk       exits, hard gates, position sizing
    executor   send sized orders to the broker
    portfolio  mark to market, persist equity

To add your own:

    from bottrade.agents import Agent, Context, register

    @register("my_agent")
    class MyAgent(Agent):
        name = "my_agent"
        async def run(self, ctx: Context) -> None:
            ...

then put `my_agent` in `[agents] enabled` (after `signal`, before `executor`,
for anything that should influence orders).
"""

from .base import Agent, Context, build_agents, get_agent_class, register, registry
from .orchestrator import Orchestrator

__all__ = [
    "Agent",
    "Context",
    "Orchestrator",
    "build_agents",
    "get_agent_class",
    "register",
    "registry",
]
