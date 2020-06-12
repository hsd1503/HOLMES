from pathlib import Path
import os
from ray.experimental.serve import BackendConfig
import ray.experimental.serve as serve
import ray

from ensemble_profiler.constants import (MODEL_SERVICE_ECG_PREFIX,
                                         AGGREGATE_PREDICTIONS,
                                         BACKEND_PREFIX,
                                         ROUTE_ADDRESS,
                                         PATIENT_NAME_PREFIX,
                                         NURSERY_ACTOR,
                                         PREDITICATE_INTERVAL)
from ensemble_profiler.store_data_actor import StatefulPatientActor
from ensemble_profiler.patient_prediction import PytorchPredictorECG
from ensemble_profiler.ensemble_predictions import Aggregate
from ensemble_profiler.ensemble_pipeline import EnsemblePipeline
import time
from ensemble_profiler.nursery import PatientActorNursery


def create_services(model_list,gpu):
    all_services = []
    # create relevant services
    model_services = []
    for i in range(len(model_list)):
        model_service_name = MODEL_SERVICE_ECG_PREFIX + "::" + str(i)
        model_services.append(model_service_name)
        serve.create_endpoint(model_service_name)
    all_services += model_services
    serve.create_endpoint(AGGREGATE_PREDICTIONS)
    all_services.append(AGGREGATE_PREDICTIONS)
    nmodel = len(model_list)
    if nmodel % gpu == 0:
        gpu_fraction = gpu / len(model_list)
    else:
        gpu_fraction = gpu / (nmodel+1)
    for service, model in zip(model_services, model_list): 
        b_config = BackendConfig(num_replicas=1, num_gpus=gpu_fraction)
        serve.create_backend(PytorchPredictorECG, BACKEND_PREFIX+service,
                             model, True, backend_config=b_config)
    serve.create_backend(Aggregate, BACKEND_PREFIX+AGGREGATE_PREDICTIONS)

    # link services to backends
    for service in all_services:
        serve.link(service, BACKEND_PREFIX+service)

    # get handles
    service_handles = {}
    for service in all_services:
        service_handles[service] = serve.get_handle(service)

    pipeline = EnsemblePipeline(model_services, service_handles)
    return pipeline, service_handles


def start_nursery():
    nursery_handle = PatientActorNursery.remote()
    ray.experimental.register_actor(NURSERY_ACTOR, nursery_handle)
    return nursery_handle


def start_patient_actors(num_patients, pipeline,
                         nursery_handle, periodic_interval=PREDITICATE_INTERVAL):
    # start actor for collecting patients_data
    actor_handles = {}
    for patient_id in range(num_patients):
        patient_name = PATIENT_NAME_PREFIX + str(patient_id)
        obj_id = nursery_handle.start_actor.remote(StatefulPatientActor,
                                                   patient_name, init_kwargs={
                                                       "patient_name":
                                                           patient_name,
                                                       "pipeline": pipeline,
                                                       "periodic_interval":
                                                           periodic_interval})
        handle = ray.get(obj_id)[0]
        actor_handles[patient_name] = handle
    return actor_handles
