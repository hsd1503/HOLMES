# HOLMES

HOLMES: Health OnLine Model Ensemble Serving for Deep Learning Models in Intensive Care Units (KDD 2020) [paper](https://arxiv.org/pdf/2008.04063.pdf)

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

##### System address

Please make sure that the configuration of url = "http://130.207.25.143:4000/jsonrpc" is changed accordingly in Serve System (serve-healthcare/latency.py) and Client System (serve_health_clients-master/sender_rpc.py)

##### Profile latency

The serve system uses profile_ensamble from serve-healthcare/latency.py

```
profiler.profile_ensemble([model], file_path, fire_clients=False)


model_list:list = list of models ,
file_path:string = path to save the result jsonl,
constraint:json = system constraint, for example {"gpu": 1, "npatient": 1},
http_host:string = host address,
fire_clients:boolean = True means the client will be fired locally, False means client will be fired from remote server,
with_data_collector:boolean = True means there will be an actor to collect the data before it's put into ensemble prediction pipeline. False means no actor to collect the data and query will be profiled directly into ensemble profile pipeline

```

##### Profile throughput

The serve system uses profile_ensamble from serve-healthcare/throughput.py

```
profiler.calculate_throughput(model_list, num_queries=300)

model_list:list = list of models ,
num_queries:int = number of queries to calculate the throughput of system
```
