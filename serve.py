import argparse

import logging
import multiprocessing

import grpc
from grpc import aio

import f5_pb2
import f5_pb2_grpc

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

import asyncio
import sys
import os
from typing import Iterable, AsyncIterable

# Kinda a hack so we don't need to maintain FS-TTS greatness
sys.path.append(os.path.join(os.path.dirname(__file__), "F5-TTS"))

from inference import TTSEngine


async def serve(server, config):
    class TTS(f5_pb2_grpc.TTSServicer):

        def __init__(self, engine: TTSEngine):
            self.e = engine
            self.lock = asyncio.Lock()

        async def Say(self, request_iterator: AsyncIterable[f5_pb2.SayMsg], context:  grpc.ServicerContext):

            async for request in request_iterator:
                async with self.lock:
                    try:
                        data = await self.e.Say(request.message)
                    except Exception as ex:
                        log.error("Error while trying to infer say: ", ex)
                        continue
                yield f5_pb2.SayResponse(data=data)

    req_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    eng = TTSEngine(req_queue, res_queue, config)
    servicer = TTS(eng)
    proc = multiprocessing.Process(target=eng.load
, args=(config.configuration,), daemon=False)
    log.info("Loading tts engine")
    proc.start()
    # Waiting for load
    res_queue.get()
    f5_pb2_grpc.add_TTSServicer_to_server(servicer, server)
    server.add_insecure_port(eng.cfg.listen_addr)
    log.info("Starting server on %s", eng.cfg.listen_addr)
    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        req_queue.put({"control": "exit"})
        proc.join(10)

def main():
    class CustomArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help(sys.stderr)
            args = {'prog': self.prog, 'message': message}
            print(f'{self.prog}: error: {message}')
            exit(2)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    parser = CustomArgParser(
        prog='serve.py',
        description='Serve the F5-TTS inference via grpc',
    )
    parser.add_argument('configuration', help="the configuration file, this should be a python file matching setup-example.py")
    config = parser.parse_args()
    log.info("Loading tts engine")

    server = aio.server()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(server,config))
    finally:
        loop.run_until_complete(server.stop(5))

if __name__ == '__main__':
     main()
