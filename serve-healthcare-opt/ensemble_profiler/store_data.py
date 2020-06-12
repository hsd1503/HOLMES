import ray
from collections import defaultdict
from ray.experimental import serve
import torch


class StorePatientData:
    """
    A Ray Serve Backend class which stores the data of every patient.
    It also passes on the data if each patients ECG values gets filled
    upto 3750 values.

    Args:
        service_handles_dict(dict[value_type, Handles]): A dictionary of 
            different service handles.
        num_queries_dict(dict[value_type,int]): A dictionary of # of queries 
            to wait for a value type. 
    """

    def __init__(self, num_queries_dict, supported_vtype="ECG"):
        self.num_queries_dict = num_queries_dict
        # store every patient data in a dictionary
        # patient_name -> { value_type: [values, ...]}
        self.patient_data = defaultdict(lambda: defaultdict(list))
        # value_type: ECG (supported right now), vitals etc.
        self.supported_vtypes = supported_vtype

    def __predicate__(self, result):
        if isinstance(result, torch.Tensor):
            return True
        return False

    def __call__(self, flask_request, info={}):
        result = ""
        # when client requests via web context
        if serve.context.web:
            patient_name = flask_request.args.get("patient_name")
            value = float(flask_request.args.get("value"))
            value_type = flask_request.args.get("vtype")
        else:
            # for profiling via kwargs
            patient_name = info["patient_name"]
            value = info["value"]
            value_type = info["vtype"]
        if value_type == self.supported_vtypes:
            # append the data point to the patient's stored data structure
            patient_val_list = self.patient_data[patient_name][value_type]
            patient_val_list.append(torch.tensor([[value]]))
            # check for prediction
            if (len(patient_val_list) == self.num_queries_dict[value_type]):
                data = torch.cat(patient_val_list, dim=1)
                data = torch.stack([data])
                result = data
                # clear the data for next queries to be recorded
                patient_val_list.clear()
            else:
                result = "Data Recorded"
        return result
