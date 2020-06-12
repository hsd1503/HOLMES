import os
import socket
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

from fire_clients import run_patient_client
from jsonrpc import JSONRPCResponseManager, dispatcher

@dispatcher.add_method
def fire_client(**kwargs):
    resp = ""
    if ("npatient" in kwargs.keys()) and ("serve_ip" in kwargs.keys()) and ("serve_port" in kwargs.keys()) and ("go_client_name" in kwargs.keys()):
        num_patients = kwargs.get("npatient")
        serve_ip = kwargs.get("serve_ip")
        serve_port = kwargs.get("serve_port")
        go_client_name = kwargs.get("go_client_name")
        waiting_time_ms = None
        if "waiting_time_ms" in kwargs:
            waiting_time_ms = kwargs.get("waiting_time_ms")

        resp += "runing valid request client={}.go npatient={}, serve_ip={}, serve_port={} ".format(go_client_name, num_patients, serve_ip, serve_port)
        server_path = serve_ip + ":" + str(serve_port)
        run_patient_client(server_path, num_patients,
                          str(go_client_name), time_ms=waiting_time_ms)
    else:
        resp += "invalid request, use default client=patient_client.go npatient=1, ip=localhost, port=8000 "
        server_path =  "localhost:8000"
        run_patient_client(server_path, 1, "patient_client")

    print(resp)
    return resp

@Request.application
def application(request):
    response = JSONRPCResponseManager.handle(
        request.data, dispatcher)
    return Response(response.json, mimetype='application/json')

if __name__ == '__main__':
    gw = os.popen("ip -4 route show default").read().split()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((gw[2], 0))
    IPv4addr = s.getsockname()[0]  #for where the server of patient_client.go request will be executed
    server_port=4000
    
    print("RPC server unning on IPv4 addr: {}".format(IPv4addr))
    run_simple(IPv4addr, server_port, application)
