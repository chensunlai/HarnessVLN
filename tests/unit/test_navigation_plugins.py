from __future__ import annotations

import asyncio

from agents import PassthroughVLNAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from memory import DummyLandmarkMemory
from schemas import NavGoal, NavTask
from vln import DummyVLNNavigator


def run(coroutine):
    return asyncio.run(coroutine)


def test_passthrough_runs_complete_vln_jobs_across_goals_without_reset() -> None:
    async def scenario():
        goals = (
            NavGoal("goal-1", "go to the far marker"),
            NavGoal("goal-2", "return to the near marker"),
        )
        task = NavTask("compound-1", goals[0], public={"goal_count": 2})
        environment = DummyNavigationEnvironment(goals, targets=(3, 1))
        navigator = DummyVLNNavigator(inference_period_s=0)
        result = await NavigationHarness(timeout_s=1).run_task(
            task,
            NavigationStack(
                PassthroughVLNAgent(), environment, vln=navigator
            ),
        )

        assert result.terminal.status == "completed"
        assert result.environment["position"] == 1
        assert result.environment["start_count"] == 1
        assert result.environment["goal_transitions"] == 1
        assert [event.name for event in result.audit].count("vln.navigate.task") == 2
        assert any(
            event.actor == "vln" and event.name == "nav.move.discrete"
            for event in result.audit
        )
        assert result.audit[-1].name == "nav.stop"

    run(scenario())


def test_local_vln_call_blocks_until_its_bounded_attempt_finishes() -> None:
    class LocalAgent:
        required_tools = frozenset({"vln.navigate.local"})

        def __init__(self):
            self.status = None

        async def run(self, context):
            self.status = await context.vln.navigate_local(
                "Move toward the visible marker ahead."
            )
            await context.nav.stop("completed", "local attempt inspected")

    async def scenario():
        goal = NavGoal("goal", "reach the marker")
        agent = LocalAgent()
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("local", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(3,)),
                vln=DummyVLNNavigator(local_max_steps=2),
            ),
        )

        assert result.terminal.status == "completed"
        assert agent.status == {
            "state": "limit_reached",
            "steps": 2,
            "reason": "step limit reached: 2",
        }
        assert result.environment["position"] == 2
        assert [event.name for event in result.audit].count("vln.navigate.local") == 1
        assert not any(
            event.name.startswith("vln.navigate.")
            and event.name != "vln.navigate.local"
            for event in result.audit
        )

    run(scenario())


def test_harness_timeout_stops_blocking_vln_call() -> None:
    async def scenario():
        goal = NavGoal("goal", "keep moving")
        navigator = DummyVLNNavigator(inference_period_s=10)
        result = await NavigationHarness(timeout_s=0.01).run_task(
            NavTask("timeout", goal),
            NavigationStack(
                PassthroughVLNAgent(),
                DummyNavigationEnvironment((goal,), targets=(100,)),
                vln=navigator,
            ),
        )

        assert result.terminal.status == "timeout"
        assert "vln.navigate.task" in [event.name for event in result.audit]
        assert navigator._jobs
        assert all(
            job.task is not None and job.task.done()
            for job in navigator._jobs.values()
        )

    run(scenario())


def test_landmarks_persist_across_new_task_and_memory_instance(tmp_path) -> None:
    class MemoryAgent:
        required_tools = frozenset({"spatial.search", "spatial.remember"})

        def __init__(self, remember=None, search=""):
            self.remember = remember
            self.search = search
            self.found = []

        async def run(self, context):
            if self.search:
                self.found = await context.spatial.search(
                    self.search,
                    frame="map",
                    near_pose=[1.0, 0.0],
                    top_k=5,
                )
            if self.remember:
                await context.spatial.remember(*self.remember)
            await context.nav.stop("completed", "memory operation complete")

    async def execute(task_id, agent, writeback=True):
        goal = NavGoal(f"{task_id}-goal", "memory test")
        return await NavigationHarness(timeout_s=1).run_task(
            NavTask(task_id, goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
                memory=DummyLandmarkMemory(tmp_path, writeback=writeback),
            ),
        )

    async def scenario():
        await execute(
            "task-a",
            MemoryAgent(remember=("kitchen doorway", "map", [2.0, 0.0])),
        )
        reader = MemoryAgent(search="kitchen")
        await execute("task-b", reader)
        assert len(reader.found) == 1
        assert reader.found[0]["source_task_id"] == "task-a"
        assert reader.found[0]["pose"] == [2.0, 0.0]

    run(scenario())


def test_writeback_false_does_not_change_existing_file(tmp_path) -> None:
    class RememberAgent:
        required_tools = frozenset({"spatial.remember"})

        async def run(self, context):
            await context.spatial.remember("temporary", "map", [0.0, 0.0])
            await context.nav.stop("completed")

    async def scenario():
        path = tmp_path / "landmarks.json"
        path.write_text("[]\n")
        before = path.read_bytes()
        goal = NavGoal("goal", "remember")
        await NavigationHarness(timeout_s=1).run_task(
            NavTask("readonly", goal),
            NavigationStack(
                RememberAgent(),
                DummyNavigationEnvironment((goal,), targets=(0,)),
                memory=DummyLandmarkMemory(tmp_path, writeback=False),
            ),
        )
        assert path.read_bytes() == before

    run(scenario())
