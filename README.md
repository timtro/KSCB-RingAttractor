# KSCB-RingAttractor
Project for the 2025 Konstanz School of Collective Behaviour.

## Building
Ensure conan is installed for dependency management.
```bash
mkdir build
cd build
../config.sh
make -j<thread_count>
```

## Dependencies
* Eigen
* `spdlog`
* ImGui
* GLFW
* GLAD
* ImPlot
* Catch2 (unit testing framework)

## Running

### Draglag dashboard

`Draglag` is a dashboard tool to explore model parameters in a live simulation with sliders. An input stimulus rotates around the ring and the lag between input and response can be seen and manipulated.
```Bash
./build/src/analysis/draglag
```
```
```

### Simulator
```bash
./build/src/simulator/simulator
```
It will connect to the visualiser and begin accepting requests.

### Visualiser (defunct)

A prior iteration of this code used ∅MQ to communicate with a python program running a dashboard with Plotly. The simulation ran far faster than the dash could update so it was not as useful as one would like, so I switched to an ImGui/ImPlot dashboard.

Create a virtual environment, perhaps under `src/visualiser/`. Activate the virtual env and then install dependencies:
```bash
python -m venv venv_new && venv/bin/python -m pip install -r requirements.txt
```
Then run
```bash
./visualiser.py
```
Open your browser to the address indicated in the `stdout` log.
