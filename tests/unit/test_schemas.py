import pytest

from schemas import NavigationTask, Terminal


def test_navigation_task_defensively_copies_metadata():
    metadata = {"scene": "one"}
    task = NavigationTask("task", "go", metadata)
    metadata["scene"] = "two"
    assert task.metadata == {"scene": "one"}


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: NavigationTask("", "go"), "task_id"),
        (lambda: NavigationTask("task", ""), "instruction"),
        (lambda: Terminal("", "reason", "agent"), "status"),
        (lambda: Terminal("completed", "reason", ""), "actor"),
    ],
)
def test_core_schema_rejects_empty_identifiers(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
