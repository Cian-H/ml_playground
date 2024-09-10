from functools import lru_cache
from pathlib import Path
import tomllib
import typer
from rich.console import Console


@lru_cache(maxsize=1)
def get(config_file: Path = Path("config.toml")) -> dict:
    with config_file.open("rb") as f:
        config = tomllib.load(f)
    config["app"] = typer.Typer()
    config["rc"] = Console()
    return config
