import os
import json
import queue
import argparse
from collections import deque
import tensorflow as tf
import numpy as np
import sounddevice as sd
import resampy
from kws_streaming.models import utils
from kws_streaming.models import model_utils
from kws_streaming.layers.modes import Modes
import kws_streaming.models.ds_tc_resnet as ds_tc_resnet

# command-line arguments parsing

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Show list of audio devices and exit", action="store_true")
parser.add_argument("-i", type=int, help="Input device index")
parser.add_argument("-o", type=int, help="Output device index")
parser.add_argument("-l", type=int, help="Latency in ms")
args = parser.parse_args()

default_samplerate = sd.query_devices(device=sd.default.device[0])['default_samplerate']
sd.default.samplerate = default_samplerate
n_blocks = 50

if args.d:
    print(sd.query_devices())
    parser.exit(0)
if args.i:
    sd.default.device[0] = args.i
if args.o:
    sd.default.device[1] = args.o
if args.l:
    latency = args.l
    n_blocks = latency / 20
    if n_blocks < 5:
        n_blocks = 5

block_size = round(default_samplerate / n_blocks)

# flags loading and model init

tf.compat.v1.disable_eager_execution()
dir = os.path.dirname(__file__)
with tf.compat.v1.gfile.Open(os.path.join(dir, 'ds_tc_resnet', 'flags.json'), 'r') as fd:
  flags_json = json.load(fd)

class DictStruct(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)

flags = DictStruct(**flags_json)
total_stride = 1

pools = model_utils.parse(flags.ds_pool)
strides = model_utils.parse(flags.ds_stride)
time_stride = [1]
for pool in pools:
  if pool > 1:
    time_stride.append(pool)
for stride in strides:
  if stride > 1:
    time_stride.append(stride)
total_stride = np.prod(time_stride)

flags.data_stride = total_stride
flags.data_shape = (total_stride * flags.window_stride_samples,)
flags.batch_size = 1

model_non_stream_batch = ds_tc_resnet.model(flags)
model_non_stream_batch.load_weights(os.path.join(dir, 'ds_tc_resnet', 'best_weights')).expect_partial()

model_stream = utils.to_streaming_inference(model_non_stream_batch, flags, Modes.STREAM_INTERNAL_STATE_INFERENCE)

q = queue.Queue()
buffer = deque()
out_q = queue.Queue()

start_idx = 0
buffered_chunks = 0

# callback for each audio block

def callback(indata, outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())
    try:
        outdata[:] = out_q.get_nowait()
    except queue.Empty:
        pass

# reading and processing audio

with sd.Stream(blocksize = block_size, channels=1, callback=callback):
    while True:
        indata = q.get()
        stream_update = indata
        stream_update[np.abs(stream_update) < 1e-8] = 0
        stream_update = np.expand_dims(stream_update.flatten(), 0)
    
        # resampling and classification of a current block
        resampled_data = resampy.resample(stream_update, block_size, 320, filter="kaiser_best")
        stream_output_prediction = model_stream.predict(resampled_data)
        stream_output_arg = np.argmax(stream_output_prediction)
        if stream_output_arg > 1:
            for i in range(5):
                buffer.popleft()
            for i in range(5):
                t = (start_idx + np.arange(block_size)) / default_samplerate
                t = t.reshape(-1, 1)
                buffer.appendleft(0.1 * np.sin(2 * np.pi * 1000 * t))
                start_idx += block_size
        if buffered_chunks < n_blocks:
            buffer.append(indata[:])
            buffered_chunks += 1
        else:
            buffer.append(indata[:])
            get = buffer.popleft()
            out_q.put(get)