
import tensorflow_transform as transform
import tensorflow as tf

_VOCAB_FEATURE_KEYS = [
    'TripStartYear',
    'TripStartMonth',
    'TripStartHour',
    'TripStartMinute',
    'pickup_census_tract',
    'dropoff_census_tract'
]

_VOCAB_SIZE = 10
_OOV_SIZE = 5

_FARE_KEY = 'fare'

_LABEL_KEY = 'fare'

_DENSE_FLOAT_FEATURE_KEYS = [
 'historical_tripDuration',
 'histOneWeek_tripDuration',
 'historical_tripDistance',
 'histOneWeek_tripDistance',
 'rawDistance',
]

_BUCKET_FEATURE_KEYS = []

_FEATURE_BUCKET_COUNT = 10

_CATEGORICAL_FEATURE_KEYS = []

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)

def _transformed_name(key):
  return key# + '_xf'

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    outputs[_transformed_name(
        key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])
    
  outputs[_transformed_name(_LABEL_KEY)] = _fill_in_missing(inputs[_FARE_KEY])

  return outputs
