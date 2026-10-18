from asyncio import Event
import queue
from queue import Queue
import time

from darker_thinner_lines import DarkerThinnerLines
from hytils import lightgreen

import torch
from torch import Tensor
from torch.cuda import StreamContext

from denc import (
    BaseThread,
    EncoderThread,
    Device,
    NnFrame,
    PipeFormat,
    dlogger,
)


class DarkerThinnerThread(BaseThread):
    def __init__(
        self,
        name: str | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._stop_event = Event()
        self._stop_event.clear()
        self.in_queue: Queue = Queue(1)
        self.verbose = debug
        self.infer_stream: torch.cuda.Stream = None


    def initialize(
        self,
        device: Device,
        pipe_format: PipeFormat,
        darken: float = 0.6,
        thin: float = 0.4,
    ):
        self.module = DarkerThinnerLines(
            darken=darken,
            thin=thin,
        )
        self.device = device
        self.pipe_format = pipe_format


    @torch.inference_mode()
    def run(self):
        verbose: bool = self.verbose
        in_queue: Queue = self.in_queue

        e_thread: EncoderThread = self.consumer

        device = self.device.name
        module = self.module.to(device=device)

        # Create a cuda stream and allocate Host memory
        cuda = bool("cuda" in self.device.name)
        if cuda:
            cuda_stream: torch.cuda.Stream = torch.cuda.Stream(device)
            stream_context: StreamContext = torch.cuda.stream(cuda_stream)
        else:
            from contextlib import nullcontext
            stream_context = nullcontext()

        with stream_context:
            while not self._stop_event.is_set():
                if verbose:
                    dlogger.debug(lightgreen("[V][I] waiting"))
                input = in_queue.get(block=True)
                if input is None or self._stop_event.is_set():
                    break
                frame: NnFrame = input

                in_tensor = frame.tensor
                # not needed to modify device/dtype, because it has been done in producer
                # which is the decoder in this case
                # in_tensor = in_tensor.to(dtype=dtype)
                d_out: Tensor = module(in_tensor)
                frame.tensor = torch.clamp(d_out, 0., 1.)

                time.sleep(0.0001)
                cuda_stream.synchronize()
                e_thread.put_frame(frame)

        while not self.in_queue.empty():
            self.in_queue.get_nowait()


    def stop(self, force: bool=False) -> None:
        self._stop_event.set()
        if force:
            while not self.in_queue.empty():
                self.in_queue.get_nowait()
        self.put_frame(None)


    def put_frame(self, frame: NnFrame) -> bool:
        try:
            self.in_queue.put(frame)
            return True
        except queue.Full:
            dlogger.debug(lightgreen("[V][I] no place"))
            pass
        return False


    def ended(self) -> bool:
        return self.in_queue.empty()
