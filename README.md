# neuromorphic-computing-project

## How to get the gym env working

0. download and install docker desktop
1. download and install: https://vcxsrv.com/
2. open it on your laptop
3. run in cmd: $env:DISPLAY="host.docker.internal:0.0"
4. run in cmd: set DISPLAY=host.docker.internal:0.0
5. clone github repo
6. docker build -t gymenv:latest .
7. docker run -it --rm -e DISPLAY=host.docker.internal:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/dri gymenv
