from ray.experimental import serve
import os
import ray
from ensemble_profiler.utils import *
import time
from ensemble_profiler.server import HTTPActor
import subprocess
from ensemble_profiler.constants import (ROUTE_ADDRESS, PROFILE_ENSEMBLE,
                                         PREDITICATE_INTERVAL)
from ensemble_profiler.tq_simulator import find_tq
import time
from threading import Event
import requests
import json
import torch
import socket
import os
import jsonlines
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))


def _calculate_throughput_ensemble(pipeline):
    num_queries = 100
    start_time = time.time()
    futures = [pipeline.remote(data=torch.zeros(1, 1, PREDITICATE_INTERVAL))
               for _ in range(num_queries)]
    result = ray.get(futures)
    end_time = time.time()
    mu_qps = num_queries / (end_time - start_time)
    return mu_qps


def _heuristic_lambda_calculation(mu_qps):
    """
    This method heuristically calculates the lambda given a throughput
    rate! Right now the heuristic is set to be 3/4th of mu_qps
    """
    return mu_qps * 0.75


def _calculate_latency(file_name, p=95):
    latency_s = []
    with jsonlines.open(file_name) as reader:
        latency_s = [(obj["end"] - obj["start"]) for obj in reader]
    return np.percentile(latency_s, p)


def profile_ensemble(model_list, file_path,
                     constraint={"gpu": 1, "npatient": 1}, http_host="0.0.0.0",
                     fire_clients=True, with_data_collector=False):
    if not ray.is_initialized():
        # read constraint
        num_patients = int(constraint["npatient"])
        gpu = int(constraint["gpu"])
        ray.init(object_store_memory=1000000000,
                 _internal_config=json.dumps(
                     {"raylet_reconstruction_timeout_milliseconds": 1000000000,
                      "initial_reconstruction_timeout_milliseconds": 1000000000}))
        serve.init(blocking=True, http_port=5000)
        nursery_handle = start_nursery()
        if not os.path.exists(str(file_path.resolve())):
            file_path.touch()
        file_name = str(file_path.resolve())
        # create the pipeline
        pipeline, service_handles = create_services(model_list, gpu)
        # create patient handles
        if with_data_collector:
            actor_handles = start_patient_actors(num_patients=num_patients,
                                                 nursery_handle=nursery_handle,
                                                 pipeline=pipeline)
        else:
            # if not data collector then only one client needed
            actor_handles = {f"patient{i}": None for i in range(1)}

        # start the http server
        obj_id = nursery_handle.start_actor.remote(HTTPActor,
                                                   "HEALTH_HTTP_SERVER",
                                                   init_args=[ROUTE_ADDRESS,
                                                              actor_handles,
                                                              pipeline,
                                                              file_name])

        http_actor_handle = ray.get(obj_id)[0]
        http_actor_handle.run.remote(host=http_host, port=8000)
        # wait for http actor to get started
        time.sleep(2)

        try:
            # warming up the gpu
            warmup_gpu(pipeline, warmup=200)

            if not with_data_collector:
                # calculating the throughput
                mu_qps = _calculate_throughput_ensemble(pipeline)
                print("Throughput of Ensemble is : {} QPS".format(mu_qps))
                lambda_qps = _heuristic_lambda_calculation(mu_qps)
                waiting_time_ms = 1000.0/lambda_qps
                print("Lambda of Ensemble is: {} QPS,"
                      " waiting time: {} ms".format(lambda_qps, waiting_time_ms))

            # fire client
            if fire_clients:
                print("Firing the clients")
                if with_data_collector:
                    client_path = os.path.join(
                        package_directory, "patient_client.go")
                    cmd = ["go", "run", client_path]
                else:
                    ensembler_path = os.path.join(
                        package_directory, "profile_ensemble.go")
                    cmd = ["go", "run", ensembler_path]

                procs = []
                for patient_name in actor_handles.keys():
                    final_cmd = cmd + [patient_name]
                    if not with_data_collector:
                        final_cmd += [str(waiting_time_ms)]
                    ls_output = subprocess.Popen(final_cmd)
                    procs.append(ls_output)
                for p in procs:
                    p.wait()
                serve.shutdown()
                T_s = _calculate_latency(file_name)
                if not with_data_collector:
                    T_q = find_tq(lambda_qps, num_patients, mu_qps, T_s)
                    return T_q + T_s
                return T_s
            else:
                gw = os.popen("ip -4 route show default").read().split()
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((gw[2], 0))
                # for where the server ray.serve() request will be executed
                IPv4addr = s.getsockname()[0]
                serve_port = 8000

                # for client address. In the experiment points to pluto
                url = "http://130.207.25.143:4000/jsonrpc"
                print("sending RPC request form IPv4 addr: {}".format(IPv4addr))
                if with_data_collector:
                    req_params = {"npatient": num_patients, "serve_ip": IPv4addr,
                                  "serve_port": serve_port, "go_client_name": "patient_client"}
                else:
                    req_params = {"npatient": 1,
                                  "serve_ip": IPv4addr,
                                  "serve_port": serve_port,
                                  "go_client_name": "profile_ensemble",
                                  "waiting_time_ms": waiting_time_ms}
                fire_remote_clients(url, req_params)
                print("finish firing remote clients")
                serve.shutdown()
                T_s = _calculate_latency(file_name)
                if not with_data_collector:
                    T_q = find_tq(lambda_qps, num_patients, mu_qps, T_s)
                    return T_q + T_s
                return T_s
        except Exception as e:
            serve.shutdown()
            print(str(e))
            return None


def fire_remote_clients(url, req_params):
    payload = {
        "method": "fire_client",
        "params": req_params,
        "jsonrpc": "2.0",
        "id": 0
    }
    response = requests.post(url, json=payload).json()
    print("{}".format(response))


def warmup_gpu(pipeline, warmup):
    print("warmup GPU")
    total_data_request = PREDITICATE_INTERVAL
    for _ in range(warmup):
        ray.get(pipeline.remote(data=torch.zeros(1, 1, total_data_request)))
    print("finish warming up GPU by firing torch zero {} times".format(warmup))
