# serve_health_clients
In order to enable running patient client please install 
[werkzeug](https://pypi.org/project/Werkzeug/)
[json-rpc](https://pypi.org/project/json-rpc/)

Make sure you've the correct address and port in the server. 
You can check that inside server_rpc.py main function
1. Run ```tmux```
2. Run RPC server inside tmux with ```$python server_rpc.py```
3. Exit tmux with CTRL+B then D. To check you can run ```tmux```
4. To login to your previous session use ```tmux a -t $tmux_id``` 
   in the case you dont have any other tmux session $tmux_id=0
