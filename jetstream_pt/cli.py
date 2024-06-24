from absl import flags, app
import os
from typing import Sequence

# import torch_xla2 first!
import torch_xla2  # pylint: disable
import jax
from absl import app, flags
from jetstream.core import server_lib
from jetstream.core.config_lib import ServerConfig, MetricsServerConfig
from jetstream_pt import fetch_models
from jetstream_pt import environment, engine


FLAGS = flags.FLAGS

flags.DEFINE_string('model_id', '', '')
flags.DEFINE_integer("override_batch_size", 32, "The batch size")
flags.DEFINE_integer("max_input_length", 1024, "The batch size")
flags.DEFINE_integer("max_output_length", 1024, "The batch size")
flags.DEFINE_string("override_datatype", 'bfloat16', "")
flags.DEFINE_integer("port", 9000, "port to listen on")
flags.DEFINE_integer("threads", 64, "number of worker threads in thread pool")

def create_engine(device):
  env_data = fetch_models.construct_env_data_from_model_id(
    FLAGS.model_id,
    FLAGS.override_batch_size,
    FLAGS.max_input_length,
    FLAGS.max_output_length,
    FLAGS.override_datatype == 'int8',
  )
  env = environment.JetEngineEnvironment(env_data)
  model, weights = fetch_models.instantiate_model_from_repo_id(FLAGS.model_id, env)
  return engine.PyTorchEngine(pt_model=model, env=env)


def list_model():
  for model_id in fetch_models.model_id_to_class.keys():
    print(model_id)


def serve():
  if FLAGS.model_id == '':
    print('Please specify model_id with --model_id')
    print('valid model ids are:')
    list_model()
    sys.exit(1)
  devices = server_lib.get_devices()
  print(f"devices: {devices}")

  server_config = ServerConfig(
      interleaved_slices=(f"tpu={len(jax.devices())}",),
      interleaved_engine_create_fns=[create_engine],
  )
  print(f"server_config: {server_config}")

  metrics_server_config: MetricsServerConfig | None = None

  # We separate credential from run so that we can unit test it with local credentials.
  # We would like to add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      threads=FLAGS.threads,
      port=FLAGS.port,
      config=server_config,
      devices=devices,
      metrics_server_config=metrics_server_config,
  )
  print("Started jetstream_server....")
  jetstream_server.wait_for_termination()

def interactive():
  raise RuntimeError("Not implemented")


def main(argv):
  if len(argv) < 2:
    print("Invalid arguments. please specify 'list' or 'serve'")

  if argv[1] == "list":
    list_model()
    return

  if argv[1] == "serve":
    serve()
    return
  
  if argv[1] == "interative":
    list_model()
    return

if __name__ == "__main__":
  app.run(main)
