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


## Running

Run the visualiser first. It will listen for the simulator to start, and when available, will poll the simulator for state updates.

### Visualiser
Create a virtual environment, perhaps under `src/visualiser/`. Activate the virtual env and then install dependencies:
```bash
python -m venv venv_new && venv/bin/python -m pip install -r requirements.txt
```
Then run
```bash
./visualiser.py
```
Open your browser to the address indicated in the `stdout` log.

### Simulator
```bash
./build/src/simulator/simulator
```
It will connect to the visualiser and begin accepting requests.
