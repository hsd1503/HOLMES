import requests
import json
import socket
import os

def fire_remote_patient(url, req_params):
    payload = {
        "method": "fire_client",
        "params": req_params,
        "jsonrpc": "2.0",
        "id": 0
    }
    response = requests.post(url, json=payload).json()
    print("{}".format(response))

if __name__ == "__main__":
    gw = os.popen("ip -4 route show default").read().split()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((gw[2], 0))
    IPv4addr = s.getsockname()[0]  #for where the server ray.serve() request will be executed
    serve_port = 8000

    url = "http://130.207.25.143:4000/jsonrpc" #for client address. In the experiment points to pluto
    print("sending RPC request form IPv4 addr: {}".format(IPv4addr))
    req_params = {"npatient":1, "serve_ip":IPv4addr, "serve_port":serve_port}
    fire_remote_patient(url, req_params)
