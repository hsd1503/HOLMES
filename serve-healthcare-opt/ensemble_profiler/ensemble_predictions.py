class Aggregate:
    """
    Perform majority voting on predicted classes.
    """
    def __init__(self):
        pass

    def __call__(self, flask_request, **kwargs_predictions):
        cnt_dict = {}
        for val in kwargs_predictions.values():
            if val in cnt_dict:
                cnt_dict[val] += 1
            else:
                cnt_dict[val] = 1
        max_cnt = 0
        for key in cnt_dict:
            if cnt_dict[key] > max_cnt:
                max_cnt = cnt_dict[key]
                vote = key
        return vote
