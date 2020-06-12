import numpy as np
import matplotlib.pyplot as P


def find_index_of_crossing(list1, list2):
    for index, (val1, val2) in enumerate(zip(list1, list2)):
        if val1 < val2:
            return index


def find_max_horizontal_distance(arrival_curve, service_curve, x_val):

    # finding the index where service curve crosses and moves
    # above the arrival curve
    crossing_index = find_index_of_crossing(arrival_curve, service_curve)

    max_distance = -1
    # horizontal distance should be calculated only when the arrival
    # curve is above the service curve
    # finding (interpolating) the x values for
    # arrival values wrt service curve values
    interpoled_vals = np.interp(arrival_curve[:crossing_index],
                                xp=service_curve, fp=x_val)
    distance_list = [(interp_val - x) for interp_val,
                     x in zip(interpoled_vals, x_val[:crossing_index])]

    # the interpolated x_values (using service curve)
    # for arrival curve should be
    # greater than x_values for service curve as arrival
    # curve is above service curve before crossing index
    max_distance = max(distance_list)
    assert max_distance > 0
    return max_distance


def find_tq(lambda_qps, num_patients, mu, T_s):
    num_queries = 500
    # dt is the waiting time for every patient
    dt = float(num_patients)/lambda_qps

    # The starting time of patient independent and comes from
    # a uniform distribution
    offsets = [np.random.uniform(0, dt) for _ in range(num_patients)]

    # collecting the arrivals of all the patients
    tstamps = []
    for cli in range(num_patients):
        arrivals = [offsets[cli] + dt*i for i in range(num_queries)]
        tstamps += arrivals
    assert (len(tstamps) == num_queries*num_patients)
    tstamps = sorted(tstamps)
    max_val = tstamps[-1]

    # plotting the service and arrival curve
    dt_window_list = []
    service_curve = []

    # starting the widow with value less than T_s
    window_size = 0.3*T_s

    # setting the range for window uptil the whole trace time
    range_val = np.log10(max_val/T_s)
    for window_offset in np.logspace(0, range_val, num=20, endpoint=True):

        # exponentially increasing the window
        window = window_offset*window_size

        # service curve doesn't need sliding window
        # just need the window size
        # the function for service curve is piece wise linear
        mu_val = mu*max((window-T_s), 0)
        service_curve.append(mu_val)

        # simulating the arrival curve
        max_num_query = -1
        for i in range(len(tstamps)):
            num_queries = 0
            j = i
            # sliding the window for arrival curve
            while j < len(tstamps) and (tstamps[j] - tstamps[i]) <= window:
                num_queries += 1
                j += 1
            max_num_query = max(max_num_query, num_queries)

        # collecting the maximum number of query got for a window
        dt_window_list.append((window, max_num_query))

    x, arrival_curve = list(zip(*dt_window_list))
    # T_q is the maximum horizontal distance b/w arrival and
    # service curve
    T_q = find_max_horizontal_distance(arrival_curve, service_curve, x)
    return T_q
