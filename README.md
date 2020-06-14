# HOLMES

HOLMES: Health OnLine Model Ensemble Serving for Deep Learning Models in Intensive Care Units (KDD 2020)

### Cloning the repository

`git clone --recurse-submodules https://github.com/hsd1503/HOLMES`

### Installing the Client System

1. `cd serve_health_clients-master`
2. Follow the README instruction to install dependencies
3. Start a client on a specific port in terminal emulation

### Installing the Serve System

1. `cd serve-healthcare`
2. Follow the README instruction to install dependencies
3. Make sure that the ip address and port of client system are configured correctly
4. To run the default example of the serve system run `python profile_example.py`

### Configuration

The serve system uses

```
profiler.profile_ensemble([model], file_path, fire_clients=False)
```
