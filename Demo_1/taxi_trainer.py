
from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_transform as tft
import tensorflow_data_validation as tfdv
from tfx import v1 as tfx_v1
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2
import keras_tuner

_FEATURE_KEYS = [
 'TripStartYear',
 'TripStartMonth',
 'TripStartHour',
 'TripStartMinute',
 'pickup_census_tract',
 'dropoff_census_tract',
 'historical_tripDuration',
 'histOneWeek_tripDuration',
 'historical_tripDistance',
 'histOneWeek_tripDistance',
 'rawDistance',]
_LABEL_KEY = 'fare'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


def _get_hyperparameters() -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = keras_tuner.HyperParameters()
    
  hp.Choice('learning_rate', [1e-2, 1e-3], default=1e-2)
  hp.Int('num_layers', 1, 3, default=2)
  return hp


def _input_fn(file_pattern: List[str],
              data_accessor: tfx_v1.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int) -> tf.data.Dataset:
  """Generates features and label for training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      schema=schema).repeat()


def _make_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for predicting taxi trips fare for the Chicago Taxi Trips assignment.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(int(hparams.get('num_layers'))):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(1)(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanSquaredError(name='rmse')])

  model.summary(print_fn=logging.info)
  return model

def tuner_fn(fn_args: tfx_v1.components.FnArgs) -> tfx_v1.components.TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  tuner = keras_tuner.RandomSearch(
      _make_keras_model,
      max_trials=6,
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=keras_tuner.Objective('rmse', 'min'),
      directory=fn_args.working_dir,
      project_name='taxi_tuning')
  schema = tfdv.load_schema_text(fn_args.schema_path)

  #transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      #transform_graph,
      _TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      #transform_graph,
      _EVAL_BATCH_SIZE)

  return tfx_v1.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })


def run_fn(fn_args: tfx_v1.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  schema = tfdv.load_schema_text(fn_args.schema_path)
  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      batch_size=_EVAL_BATCH_SIZE)

  model = _make_keras_model({'num_layers': 3, 'learning_rate': 1e-2})
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  model.save(fn_args.serving_model_dir, save_format='tf')
