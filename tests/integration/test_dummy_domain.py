import asyncio

from agents.dummy import DummyAgent
from benches.dummy import DummyBenchmark
from envs.dummy import DummyEnvironment
from harness import DomainComponents, DomainRuntime
from memory.dummy import DummyMemory
from metrics.dummy import DummyMetric
from vln.dummy import DummyVLN


def test_dummy_components_form_a_complete_agent_led_navigation_stack(tmp_path):
    async def scenario():
        results = []
        benchmark = DummyBenchmark()
        for case in benchmark.cases():
            components = DomainComponents(
                environment=DummyEnvironment(),
                agent=DummyAgent(),
                services=(DummyMemory(), DummyVLN()),
                metrics=(DummyMetric(),),
            )
            result = await DomainRuntime(timeout_s=2).run(
                case.task,
                components,
                output_root=str(tmp_path),
                domain_id=case.case_id,
            )
            results.append(result)

        assert all(result.terminal.status == "completed" for result in results)
        assert [result.environment["position"] for result in results] == [2, -3, 0]
        assert benchmark.aggregate(results) == {
            "success": 1.0,
            "spl": 1.0,
            "distance": 0.0,
        }
        return results

    results = asyncio.run(scenario())
    first = tmp_path / results[0].domain_id
    assert (first / "components" / "environment" / "trajectory.json").is_file()
    assert (first / "components" / "agent" / "model" / "trace.jsonl").is_file()
    assert (first / "components" / "vln" / "inference" / "trace.jsonl").is_file()
    assert (first / "components" / "memory" / "memory.json").is_file()
    assert (first / "components" / "metric" / "metrics.json").is_file()
