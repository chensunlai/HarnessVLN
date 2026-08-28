from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_implementations_live_in_named_third_level_packages():
    for category in ("agents", "benches", "envs", "memory", "metrics", "vln"):
        directory = ROOT / "src" / category
        assert (directory / "__init__.py").is_file()
        implementations = [
            path
            for path in directory.iterdir()
            if path.is_dir() and not path.name.startswith("__")
        ]
        assert implementations
        assert all((path / "__init__.py").is_file() for path in implementations)


def test_core_does_not_import_concrete_components():
    core = "\n".join(
        path.read_text()
        for path in (ROOT / "src" / "harness").glob("*.py")
    )
    for category in ("agents.", "envs.", "memory.", "metrics.", "vln."):
        assert f"import {category}" not in core
        assert f"from {category}" not in core
