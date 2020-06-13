# serve_health_clients
### Install Golang
Installation details can be found [here](https://golang.org/doc/install). 

### Installing Python Packages
1. `pip install -r requirement.txt`
In order to enable running patient client please install 
[werkzeug](https://pypi.org/project/Werkzeug/)
[json-rpc](https://pypi.org/project/json-rpc/)

### Running Client on Background
Make sure you've the correct ip address and port in the server. 
You can check ip address and port inside server_rpc.py main function
1. Run ```tmux```
2. Run RPC server inside tmux with ```$python server_rpc.py```
3. Exit tmux with CTRL+B then D. To check you can run ```tmux```
4. To login to your previous session use ```tmux a -t $tmux_id``` 
   in the case you dont have any other tmux session $tmux_id=0

In case there's no tmux you can also use ``screen`` or other alternatives for screen manager terminal emulation
