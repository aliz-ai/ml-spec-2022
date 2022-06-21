import tensorflow_model_analysis as tfma
import tensorflow as tf
from tfx import v1 as tfx_v1
import tfx
import kfp

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str, project_id: str,
                     endpoint_name: str, region: str,
                     ) -> tfx_v1.dsl.Pipeline:
  """Defines a pipeline for the Chicago Taxi Trips assignment"""

  input = tfx_v1.proto.Input(splits=[
                tfx.proto.example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                tfx.proto.example_gen_pb2.Input.Split(name='eval', pattern='eval/*'),
                tfx.proto.example_gen_pb2.Input.Split(name='test', pattern='test/*')
            ])
  example_gen = tfx_v1.components.CsvExampleGen(input_base=data_root, input_config=input)

  compute_eval_stats = tfx_v1.components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      )
  schema_gen = tfx_v1.components.SchemaGen(
    statistics=compute_eval_stats.outputs['statistics'])
    
  validate_stats = tfx_v1.components.ExampleValidator(
      statistics=compute_eval_stats.outputs['statistics'],
      schema=schema_gen.outputs['schema']
      )

  transform = tfx_v1.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=preprocess_module_file)
    
  vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
          },
      }],
  }

  tuner = tfx_v1.components.Tuner(
    module_file=module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=tfx_v1.proto.TrainArgs(num_steps=20),
    schema=schema_gen.outputs['schema'],
    eval_args=tfx_v1.proto.EvalArgs(num_steps=5))
    
    
  trainer = tfx.v1.extensions.google_cloud_ai_platform.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      hyperparameters=tuner.outputs['best_hyperparameters'],
      train_args=tfx_v1.proto.TrainArgs(num_steps=100),
      eval_args=tfx_v1.proto.EvalArgs(num_steps=5),
      custom_config={
          tfx_v1.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx_v1.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx_v1.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
              vertex_job_spec,
          'use_gpu':
              False,
      })


  eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(label_key='fare')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='MeanSquaredError'),
                tfma.MetricConfig(
                    class_name='MeanSquaredError',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            upper_bound={'value': 50}),))
            ]
        )
    ],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['TripStartHour'])
    ])



  model_analyzer = tfx_v1.components.Evaluator(
      examples=transform.outputs['transformed_examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config)

  vertex_serving_spec = {
      'project_id': project_id,
      'endpoint_name': endpoint_name,
      'machine_type': 'n1-standard-4',
  }

  serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
  pusher = tfx_v1.extensions.google_cloud_ai_platform.Pusher(
      model=trainer.outputs['model'],
      custom_config={
          tfx_v1.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx_v1.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx_v1.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
              serving_image,
          tfx_v1.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
            vertex_serving_spec,
      })

  components = [
      example_gen,
      compute_eval_stats,
      schema_gen,
      validate_stats,
      transform,
      tuner,
      trainer,
      model_analyzer,
      pusher,
  ]

  return tfx_v1.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,)