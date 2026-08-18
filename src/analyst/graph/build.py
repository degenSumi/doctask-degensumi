"""Assembles the stages into a graph, with a checkpointer behind it.

The graph is where the two structural behaviours come from. Stages are named and
their transitions are explicit, so a run can be watched rather than inferred.
And every stage boundary is a checkpoint, so a process killed in the middle
resumes from the last completed stage instead of the beginning.

The human gate is an interrupt before `commit`. The graph genuinely stops there:
the run ends, state is on disk, and nothing reaches the register until decisions
are supplied and the graph is resumed. That is the same mechanism whether the
decisions come from a person at a terminal or from another program.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from analyst.graph.nodes import Nodes
from analyst.graph.state import RunState
from analyst.ports.language_model import LanguageModel


def build_graph(model: LanguageModel) -> StateGraph[RunState, None, RunState, RunState]:
    """Wire the stages together.

    `extract` loops back to itself when a document asked to be read again, which
    is the retry path. Everything else runs in order.
    """
    nodes = Nodes(model)
    graph: StateGraph[RunState, None, RunState, RunState] = StateGraph(RunState)

    graph.add_node("ingest", nodes.ingest)
    graph.add_node("classify", nodes.classify)
    graph.add_node("extract", nodes.extract)
    graph.add_node("reconcile", nodes.reconcile)
    graph.add_node("examine", nodes.examine)
    graph.add_node("propose", nodes.propose)
    graph.add_node("commit", nodes.commit)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "classify")
    graph.add_edge("classify", "extract")

    # The one conditional edge: a document that yielded nothing from its opening
    # passage is read again in full before the run moves on.
    graph.add_conditional_edges(
        "extract",
        _route_after_extract,
        {"extract": "extract", "reconcile": "reconcile"},
    )

    graph.add_edge("reconcile", "examine")
    graph.add_edge("examine", "propose")
    graph.add_edge("propose", "commit")
    graph.add_edge("commit", END)

    return graph


def _route_after_extract(state: RunState) -> str:
    """Send a run back through extraction while any document has a retry pending.

    The retry marker is raised to 2 once used, so a document is read again at
    most once and a run cannot loop.
    """
    retries = state.get("retries") or {}
    return "extract" if any(int(v) == 1 for v in retries.values()) else "reconcile"


def compile_graph(
    model: LanguageModel, checkpoint_path: Path | str = "runs.db"
) -> tuple[Any, sqlite3.Connection]:
    """Compile the graph with a checkpointer and the human gate in place.

    Run state is checkpointed to SQLite, so a fresh clone runs with no database
    to install. The connection is returned alongside so the caller can close it.
    """
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    saver = SqliteSaver(connection)

    compiled = build_graph(model).compile(
        checkpointer=saver,
        # The run stops here. Nothing is written to the register until a person
        # has decided on every item and the run is resumed.
        interrupt_before=["commit"],
    )
    return compiled, connection
