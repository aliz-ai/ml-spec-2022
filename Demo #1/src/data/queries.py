QUERY_SKELETON = """EXPORT DATA
  OPTIONS (
    uri = 'gs://aliz-ml-spec-2022/demo-1/data/taxi_data/{split}/taxi_*.csv',
    format = 'CSV',
    overwrite = true,
    header = true,
    field_delimiter = ',')
AS (
  SELECT
    CAST(TripStartYear AS STRING) AS TripStartYear,
    CAST(TripStartMonth AS STRING) AS TripStartMonth,
    CAST(TripStartHour AS STRING) AS TripStartHour,
    CAST(TripStartMinute AS STRING) AS TripStartMinute,
    CAST(pickup_census_tract AS STRING) AS pickup_census_tract,
    CAST(dropoff_census_tract AS STRING) AS dropoff_census_tract,
    IFNULL(fare, 0) AS fare,
    IFNULL(historical_tripDuration, 0) AS historical_tripDuration,
    IFNULL(histOneWeek_tripDuration, 0) AS histOneWeek_tripDuration,
    IFNULL(historical_tripDistance, 0) AS historical_tripDistance,
    IFNULL(histOneWeek_tripDistance, 0) AS histOneWeek_tripDistance,
    IFNULL(rawDistance, 0) AS rawDistance, 
FROM `aliz-ml-spec-2022-submission.demo1.Demo1_MLdataset`
{filter}
);"""


TRAIN_QUERY = QUERY_SKELETON.format(split="train", 
                                    filter = """WHERE trip_start_timestamp >= '2021-01-01 00:00:00 UTC'
    AND trip_start_timestamp < '2022-01-01 00:00:00 UTC'
    AND ABS(MOD(FARM_FINGERPRINT(trip_id), 20)) != 0""")


EVAL_QUERY = QUERY_SKELETON.format(split="eval", 
                                    filter = """WHERE trip_start_timestamp >= '2021-01-01 00:00:00 UTC'
    AND trip_start_timestamp < '2022-01-01 00:00:00 UTC'
    AND ABS(MOD(FARM_FINGERPRINT(trip_id), 20)) = 0""")

TEST_QUERY = QUERY_SKELETON.format(split="test", 
                                    filter = """WHERE trip_start_timestamp >= '2022-01-01 00:00:00 UTC'""")
