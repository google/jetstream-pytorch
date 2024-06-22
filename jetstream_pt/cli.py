from absl import flags

flags.DEFINE_string('model_id', '', '')
flags.DEFINE_integer("override_batch_size", 32, "The batch size")
flags.DEFINE_integer("max_input_length", 1024, "The batch size")
flags.DEFINE_integer("max_output_length", 1024, "The batch size")
flags.DEFINE_string("override_datatype", 'bfloat16', "")

def create_engine():
  env_data = fetch_models.construct_env_data_from_model_id(
    FLAGS.model_id,
    FLAGS.override_batch_size,
    FLAGS.max_input_length,
    FLAGS.max_output_length,
    FLAGS.override_datatype,
  )
  env = Enviroment(env_data)
  model = fetch_models.instantiate_model_from_repo_id(FLAGS.model_id, env)
  return PyTorchEngine(pt_model=pt_model, env=env)


def main(argv: Sequence[str]):
  print(argv)
  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  print(f"devices: {devices}")

  server_config = ServerConfig(
      interleaved_slices=(f"tpu={len(jax.devices())}",),
      interleaved_engine_create_fns=create_engine),
  )
  print(f"server_config: {server_config}")

  metrics_server_config: MetricsServerConfig | None = None
  if FLAGS.prometheus_port != 0:
    metrics_server_config = MetricsServerConfig(port=FLAGS.prometheus_port)

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


if __name__ == "__main__":
  app.run(main)
