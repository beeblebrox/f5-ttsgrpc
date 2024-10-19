# f5-ttsgrpc
Utilizes https://github.com/SWivid/F5-TTS,

This is a simple grpc tts server that can run inference for a single voice and prompt at a time. Each generation is returned back as bytes of a wav pcm file.

This is designed to be used in a pipeline with multiple instances ran with different configurations of voices and ports.

The server probably can handle a few client connections at a time. Voice generation is serialized between client connections so the more clients the more lag between generations may be experienced.

## Future plans/Wish list (please contribute)
* potentially a docker container with multiple instances
* example integration
* a plexer that wraps all of the python processes and exposes a hgher level abstraction via grpc that will support multiple voices. (go implementation)
  
## GRPC Services
See [protobuf definition](f5.proto)

## User Installation
### Download and init repository
```shell
git clone https://github.com/beeblebrox/ff5-ttsgrpc
git submodule update --recursive
```

### Setup python 3.10 conda environment, or your favorite env manager

## Install requirements
```shell
pip install -r requirements.txt
```

## Dev Information
### Generate grpc
From the base directory after venv is setup and sourced:

```bash
python generateproto.py
```

### IDE info
Make sure you add the directory <projectroot>/F5-TTS to your system path or marked as source in project.
#### PyCharm
![img.png](doc/pycharmaddsource.png)
