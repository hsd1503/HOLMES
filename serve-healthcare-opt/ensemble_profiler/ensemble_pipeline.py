import ray
import ray.experimental.serve as serve
from ensemble_profiler.constants import AGGREGATE_PREDICTIONS


class EnsemblePipeline:
    def __init__(self, model_services, service_handles):
        self.model_services = model_services
        self.service_handles = service_handles

    def remote(self, data):
        kwargs_for_aggregate = {}
        for model_service in self.model_services:
            md_object_id = ray.ObjectID.from_random()
            kwargs_for_aggregate[model_service] = md_object_id
            self.service_handles[model_service].remote(
                data=data,
                return_object_ids={serve.RESULT_KEY: md_object_id}
            )
        aggregate_object_id = ray.ObjectID.from_random()
        self.service_handles[AGGREGATE_PREDICTIONS].remote(
            **kwargs_for_aggregate,
            return_object_ids={serve.RESULT_KEY: aggregate_object_id}
        )
        return aggregate_object_id
