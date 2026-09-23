# Swarm Hitting Time Simulator

Simulates robots moving on a hex-oriented grid and records the number of
timesteps required to find the central target. The core uses sparse transition
matrices, so larger arenas do not require dense `6N² × 6N²` matrices.

## Project layout

| Location | Purpose |
| --- | --- |
| `Scripts/swarm_sim/` | Simulation package: arena construction, movement, state conversion, statistics, and CSV output. |
| `Scripts/run_swarm_simulation.py` | Command-line and Visual Studio startup entry point. |
| `Scripts/tests/` | Pytest suite and fixed regression fixtures. |
| `Data/` | Generated simulation CSVs, grouped by run parameters. |
| `Scripts/Graphers/` | Optional plotting and analysis utilities. |

## Run from a terminal

Use Python 3.10 or later. From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Scripts\run_swarm_simulation.py --grid-size 5 --robots 2 --iterations 10
```

Results are appended to a parameter-specific CSV below `Data/`. The default
mode waits for all robots to hit the target. For a small first-arrival run:

```powershell
python Scripts\run_swarm_simulation.py --grid-size 5 --robots 2 --iterations 10 --first-hit
```

See all run options with `python Scripts\run_swarm_simulation.py --help`.

## Test

```powershell
python -m pytest Scripts\tests -q
```

## Visual Studio solution

1. Install Visual Studio 2022 with the **Python development** workload.
2. Open `SwarmHittingTimeSimulator.sln`.
3. In **View → Other Windows → Python Environments**, add or select
   `.venv\Scripts\python.exe` as the project interpreter.
4. Restore dependencies from the Visual Studio terminal with
   `python -m pip install -r requirements.txt`.
5. Set **SwarmHittingTimeSimulator** as the startup project, then press `F5`.
   The startup file is `Scripts/run_swarm_simulation.py`; update its launch
   arguments under **Project → Properties → Debug** when needed.
6. Run the tests from **Test Explorer**, or execute the test command above.

Local Visual Studio state (`.vs/` and `*.pyproj.user`) is intentionally
excluded from source control.
