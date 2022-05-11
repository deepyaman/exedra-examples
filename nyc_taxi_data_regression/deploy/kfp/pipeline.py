from pathlib import Path

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
)


@component(
    # https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments#scikit-learn
    packages_to_install=['scikit-learn==0.24.1', 'numpy>=1.16.0', 'pandas~=1.1.0'],
    output_component_file=str(Path(__file__).parent / 'prep.yaml')
)
def prep(
    raw_green_data: Input[Dataset],
    raw_yellow_data: Input[Dataset],
    green_prep_data: Output[Dataset],
    yellow_prep_data: Output[Dataset],
    merged_data: Output[Dataset]
):
  import pandas as pd

  # Prep the green and yellow taxi data
  print("reading file: %s ..." % raw_green_data.path)
  green_data = pd.read_csv(raw_green_data.path)
  print("reading file: %s ..." % raw_yellow_data.path)
  yellow_data = pd.read_csv(raw_yellow_data.path)

  # Define useful columns needed for the Azure Machine Learning NYC Taxi tutorial

  useful_columns = str(
      [
          "cost",
          "distance",
          "dropoff_datetime",
          "dropoff_latitude",
          "dropoff_longitude",
          "passengers",
          "pickup_datetime",
          "pickup_latitude",
          "pickup_longitude",
          "store_forward",
          "vendor",
      ]
  ).replace(",", ";")
  print(useful_columns)

  # Rename columns as per Azure Machine Learning NYC Taxi tutorial
  green_columns = str(
      {
          "vendorID": "vendor",
          "lpepPickupDatetime": "pickup_datetime",
          "lpepDropoffDatetime": "dropoff_datetime",
          "storeAndFwdFlag": "store_forward",
          "pickupLongitude": "pickup_longitude",
          "pickupLatitude": "pickup_latitude",
          "dropoffLongitude": "dropoff_longitude",
          "dropoffLatitude": "dropoff_latitude",
          "passengerCount": "passengers",
          "fareAmount": "cost",
          "tripDistance": "distance",
      }
  ).replace(",", ";")

  yellow_columns = str(
      {
          "vendorID": "vendor",
          "tpepPickupDateTime": "pickup_datetime",
          "tpepDropoffDateTime": "dropoff_datetime",
          "storeAndFwdFlag": "store_forward",
          "startLon": "pickup_longitude",
          "startLat": "pickup_latitude",
          "endLon": "dropoff_longitude",
          "endLat": "dropoff_latitude",
          "passengerCount": "passengers",
          "fareAmount": "cost",
          "tripDistance": "distance",
      }
  ).replace(",", ";")

  print("green_columns: " + green_columns)
  print("yellow_columns: " + yellow_columns)

  # These functions ensure that null data is removed from the dataset,
  # which will help increase machine learning model accuracy.


  def get_dict(dict_str):
      pairs = dict_str.strip("{}").split(";")
      new_dict = {}
      for pair in pairs:
          print(pair)
          key, value = pair.strip().split(":")
          new_dict[key.strip().strip("'")] = value.strip().strip("'")
      return new_dict


  def cleanseData(data, columns, useful_columns):
      useful_columns = [
          s.strip().strip("'") for s in useful_columns.strip("[]").split(";")
      ]
      new_columns = get_dict(columns)

      new_df = (data.dropna(how="all").rename(columns=new_columns))[useful_columns]

      new_df.reset_index(inplace=True, drop=True)
      return new_df


  green_data_clean = cleanseData(green_data, green_columns, useful_columns)
  yellow_data_clean = cleanseData(yellow_data, yellow_columns, useful_columns)

  # Append yellow data to green data
  combined_df = green_data_clean.append(yellow_data_clean, ignore_index=True)
  combined_df.reset_index(inplace=True, drop=True)

  output_green = green_data_clean.to_csv(green_prep_data.path)
  output_yellow = yellow_data_clean.to_csv(
      yellow_prep_data.path
  )
  merged_data = combined_df.to_csv(merged_data.path)


@component(
    # https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments#scikit-learn
    packages_to_install=['scikit-learn==0.24.1', 'numpy>=1.16.0', 'pandas~=1.1.0'],
    output_component_file=str(Path(__file__).parent / 'transform.yaml')
)
def transform(clean_data: Input[Dataset], transformed_data: Output[Dataset]):
  import numpy as np
  import pandas as pd

  # Transform the data
  print("reading file: %s ..." % clean_data.path)
  combined_df = pd.read_csv(clean_data.path)
  # These functions filter out coordinates for locations that are outside the city border.

  # Filter out coordinates for locations that are outside the city border.
  # Chain the column filter commands within the filter() function
  # and define the minimum and maximum bounds for each field

  combined_df = combined_df.astype(
      {
          "pickup_longitude": "float64",
          "pickup_latitude": "float64",
          "dropoff_longitude": "float64",
          "dropoff_latitude": "float64",
      }
  )

  latlong_filtered_df = combined_df[
      (combined_df.pickup_longitude <= -73.72)
      & (combined_df.pickup_longitude >= -74.09)
      & (combined_df.pickup_latitude <= 40.88)
      & (combined_df.pickup_latitude >= 40.53)
      & (combined_df.dropoff_longitude <= -73.72)
      & (combined_df.dropoff_longitude >= -74.72)
      & (combined_df.dropoff_latitude <= 40.88)
      & (combined_df.dropoff_latitude >= 40.53)
  ]

  latlong_filtered_df.reset_index(inplace=True, drop=True)

  # These functions replace undefined values and rename to use meaningful names.
  replaced_stfor_vals_df = latlong_filtered_df.replace(
      {"store_forward": "0"}, {"store_forward": "N"}
  ).fillna({"store_forward": "N"})

  replaced_distance_vals_df = replaced_stfor_vals_df.replace(
      {"distance": ".00"}, {"distance": 0}
  ).fillna({"distance": 0})

  normalized_df = replaced_distance_vals_df.astype({"distance": "float64"})

  # These functions transform the renamed data to be used finally for training.

  # Split the pickup and dropoff date further into the day of the week, day of the month, and month values.
  # To get the day of the week value, use the derive_column_by_example() function.
  # The function takes an array parameter of example objects that define the input data,
  # and the preferred output. The function automatically determines your preferred transformation.
  # For the pickup and dropoff time columns, split the time into the hour, minute, and second by using
  # the split_column_by_example() function with no example parameter. After you generate the new features,
  # use the drop_columns() function to delete the original fields as the newly generated features are preferred.
  # Rename the rest of the fields to use meaningful descriptions.

  temp = pd.DatetimeIndex(normalized_df["pickup_datetime"], dtype="datetime64[ns]")
  normalized_df["pickup_date"] = temp.date
  normalized_df["pickup_weekday"] = temp.dayofweek
  normalized_df["pickup_month"] = temp.month
  normalized_df["pickup_monthday"] = temp.day
  normalized_df["pickup_time"] = temp.time
  normalized_df["pickup_hour"] = temp.hour
  normalized_df["pickup_minute"] = temp.minute
  normalized_df["pickup_second"] = temp.second

  temp = pd.DatetimeIndex(normalized_df["dropoff_datetime"], dtype="datetime64[ns]")
  normalized_df["dropoff_date"] = temp.date
  normalized_df["dropoff_weekday"] = temp.dayofweek
  normalized_df["dropoff_month"] = temp.month
  normalized_df["dropoff_monthday"] = temp.day
  normalized_df["dropoff_time"] = temp.time
  normalized_df["dropoff_hour"] = temp.hour
  normalized_df["dropoff_minute"] = temp.minute
  normalized_df["dropoff_second"] = temp.second

  del normalized_df["pickup_datetime"]
  del normalized_df["dropoff_datetime"]

  normalized_df.reset_index(inplace=True, drop=True)


  print(normalized_df.head)
  print(normalized_df.dtypes)


  # Drop the pickup_date, dropoff_date, pickup_time, dropoff_time columns because they're
  # no longer needed (granular time features like hour,
  # minute and second are more useful for model training).
  del normalized_df["pickup_date"]
  del normalized_df["dropoff_date"]
  del normalized_df["pickup_time"]
  del normalized_df["dropoff_time"]

  # Change the store_forward column to binary values
  normalized_df["store_forward"] = np.where((normalized_df.store_forward == "N"), 0, 1)

  # Before you package the dataset, run two final filters on the dataset.
  # To eliminate incorrectly captured data points,
  # filter the dataset on records where both the cost and distance variable values are greater than zero.
  # This step will significantly improve machine learning model accuracy,
  # because data points with a zero cost or distance represent major outliers that throw off prediction accuracy.

  final_df = normalized_df[(normalized_df.distance > 0) & (normalized_df.cost > 0)]
  final_df.reset_index(inplace=True, drop=True)
  print(final_df.head)

  # Output data
  transformed_data = final_df.to_csv(transformed_data.path)


@component(
    # https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments#scikit-learn
    packages_to_install=['scikit-learn==0.24.1', 'numpy>=1.16.0', 'pandas~=1.1.0'],
    output_component_file=str(Path(__file__).parent / 'train.yaml')
)
def train(
    training_data: Input[Dataset],
    test_data: Output[Dataset],
    model_output: Output[Model]
):
  import pickle

  import pandas as pd
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split

  print("reading file: %s ..." % training_data.path)
  train_data = pd.read_csv(training_data.path)
  print(train_data.columns)

  # Split the data into input(X) and output(y)
  y = train_data["cost"]
  # X = train_data.drop(['cost'], axis=1)
  X = train_data[
      [
          "distance",
          "dropoff_latitude",
          "dropoff_longitude",
          "passengers",
          "pickup_latitude",
          "pickup_longitude",
          "store_forward",
          "vendor",
          "pickup_weekday",
          "pickup_month",
          "pickup_monthday",
          "pickup_hour",
          "pickup_minute",
          "pickup_second",
          "dropoff_weekday",
          "dropoff_month",
          "dropoff_monthday",
          "dropoff_hour",
          "dropoff_minute",
          "dropoff_second",
      ]
  ]

  # Split the data into train and test sets
  trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)
  print(trainX.shape)
  print(trainX.columns)

  # Train a Linear Regression Model with the train set
  model = LinearRegression().fit(trainX, trainy)
  print(model.score(trainX, trainy))


  # Output the model and test data
  pickle.dump(model, open(model_output.path, "wb"))
  # Model artifact has a `.metadata` dictionary
  # to store arbitrary metadata for the output artifact.
  model_output.metadata['r2'] = model.score(trainX, trainy)
  # test_data = pd.DataFrame(testX, columns = )
  testX["cost"] = testy
  print(testX.shape)
  test_data = testX.to_csv(test_data.path)


@component(
    # https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments#scikit-learn
    packages_to_install=['scikit-learn==0.24.1', 'numpy>=1.16.0', 'pandas~=1.1.0'],
    output_component_file=str(Path(__file__).parent / 'predict.yaml')
)
def predict(
    model_input: Input[Model],
    test_data: Input[Dataset],
    predictions: Output[Dataset]
):
  import pickle

  import pandas as pd

  print("reading file: %s ..." % test_data.path)
  test_data = pd.read_csv(test_data.path)
  testy = test_data["cost"]
  # testX = test_data.drop(['cost'], axis=1)
  testX = test_data[
      [
          "distance",
          "dropoff_latitude",
          "dropoff_longitude",
          "passengers",
          "pickup_latitude",
          "pickup_longitude",
          "store_forward",
          "vendor",
          "pickup_weekday",
          "pickup_month",
          "pickup_monthday",
          "pickup_hour",
          "pickup_minute",
          "pickup_second",
          "dropoff_weekday",
          "dropoff_month",
          "dropoff_monthday",
          "dropoff_hour",
          "dropoff_minute",
          "dropoff_second",
      ]
  ]
  print(testX.shape)
  print(testX.columns)

  # Load the model from input port
  model = pickle.load(open(model_input.path, "rb"))
  # model = (Path(args.model_input) / 'model.txt').read_text()
  # print('Model: ', model)

  # Make predictions on testX data and record them in a column named predicted_cost
  testX["predicted_cost"] = model.predict(testX)
  print(testX.shape)

  # Compare predictions to actuals (testy)
  output_data = pd.DataFrame(testX)
  output_data["actual_cost"] = testy


  # Save the output data with feature columns, predicted cost, and actual cost in csv file
  output_data = output_data.to_csv(predictions.path)


@component(
    # https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments#scikit-learn
    packages_to_install=['scikit-learn==0.24.1', 'numpy>=1.16.0', 'pandas~=1.1.0'],
    output_component_file=str(Path(__file__).parent / 'score.yaml')
)
def score(
    predictions: Input[Dataset],
    model: Input[Model],
    score_report: Output[Artifact]
):
  import pickle
  from pathlib import Path

  import pandas as pd
  from sklearn.metrics import mean_squared_error, r2_score

  print("reading file: %s ..." % predictions.path)
  test_data = pd.read_csv(predictions.path)

  # Load the model from input port
  model = pickle.load(open(model.path, "rb"))

  # Print the results of scoring the predictions against actual values in the test data
  # The coefficients
  print("Coefficients: \n", model.coef_)

  actuals = test_data["actual_cost"]
  predictions = test_data["predicted_cost"]

  # The mean squared error
  print("Mean squared error: %.2f" % mean_squared_error(actuals, predictions))
  # The coefficient of determination: 1 is perfect prediction
  print("Coefficient of determination: %.2f" % r2_score(actuals, predictions))
  print("Model: ", model)

  # Print score report to a text file
  Path(score_report.path).write_text(
      "Scored with the following model:\n{}".format(model)
  )
  with open(score_report.path, "a") as f:
      f.write("\n Coefficients: \n %s \n" % str(model.coef_))
      f.write("Mean squared error: %.2f \n" % mean_squared_error(actuals, predictions))
      f.write("Coefficient of determination: %.2f \n" % r2_score(actuals, predictions))



# TODO(deepyaman): Leverage `importer` (not supported with v1 compiler).
# https://github.com/kubeflow/pipelines/blob/master/samples/v2/pipeline_with_importer.py
web_downloader_op = kfp.components.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/contrib/web/Download/component-sdk-v2.yaml')


@dsl.pipeline(
    name='nyc-taxi-data-regression',
    # You can optionally specify your own pipeline_root
    # pipeline_root='gs://my-pipeline-root/example-pipeline',
)
def nyc_taxi_data_regression_pipeline(raw_green_data_url: str, raw_yellow_data_url: str):
  raw_green_data_web_downloader_task = web_downloader_op(url=raw_green_data_url)
  raw_yellow_data_web_downloader_task = web_downloader_op(url=raw_yellow_data_url)
  prep_task = prep(
    raw_green_data=raw_green_data_web_downloader_task.outputs['data'],
    raw_yellow_data=raw_yellow_data_web_downloader_task.outputs['data'],
  )
  transform_task = transform(clean_data=prep_task.outputs['merged_data'])
  train_task = train(training_data=transform_task.outputs['transformed_data'])
  predict_task = predict(
    model_input=train_task.outputs['model_output'],
    test_data=train_task.outputs['test_data'],
  )
  score_task = score(
    predictions=predict_task.outputs['predictions'],
    model=train_task.outputs['model_output'],
  )


if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
        pipeline_func=nyc_taxi_data_regression_pipeline,
        package_path=str(Path(__file__).parent / 'pipeline.yaml'))
