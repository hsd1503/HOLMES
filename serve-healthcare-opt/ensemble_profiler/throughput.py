from ensemble_profiler.utils import *
import ray
from ray.experimental import serve
from ensemble_profiler.constants import PATIENT_NAME_PREFIX
import time


def calculate_throughput(model_list, num_queries=300):
    if not ray.is_initialized():
        serve.init(blocking=True)
        nursery_handle = start_nursery()
        pipeline = create_services(model_list)

        actor_handles = start_patient_actors(num_patients=1,
                                             nursery_handle=nursery_handle,
                                             pipeline=pipeline)
        patient_handle = list(actor_handles.values())[0]

        future_list = []

        # dummy request
        info = {
            "patient_name": PATIENT_NAME_PREFIX + str(0),
            "value": 1.0,
            "vtype": "ECG"
        }
        start_time = time.time()
        for _ in range(num_queries):
            fut = patient_handle.get_periodic_predictions.remote(info=info)
            future_list.append(fut)
        ray.get(future_list)
        end_time = time.time()
        serve.shutdown()
        return end_time - start_time, num_queries
