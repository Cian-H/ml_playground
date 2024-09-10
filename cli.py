#!/usr/bin/env -S rye run python

from pathlib import Path
import pandas as pd
from rich.markdown import Markdown

import config

app = config.get()["app"]
__rc = config.get()["rc"]

experiments = Path(__file__).parent.glob("expt*.py")
run_funcs = dict()

for f in experiments:
    m = __import__(f.stem)
    app.command(name=m.__name__)(m.run)
    run_funcs[m.__name__] = m.run


@app.command()
def run(expts: str = "", md: bool = False):
    to_run = run_funcs.copy()
    if expts != "":
        expts = expts.split(",")
        to_run = {k: v for k, v in to_run.items() if k in expts}
        to_run = dict(sorted(to_run.items(), key=lambda kv: expts.index(kv[0])))
    results = []
    for k, v in to_run.items():
        print(f"Running {k}...")
        r = v()
        r["Experiment"] = k
        results.append(r)
    results = pd.concat(results)
    cols = list(results.keys())
    cols.remove("Experiment")
    cols = ["Experiment", *cols]
    results = results[cols].reset_index(drop=True)
    r = results.to_markdown()
    if md:
        print(r)
    else:
        __rc.print(Markdown(r))


if __name__ == "__main__":
    app()
