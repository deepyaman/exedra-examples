from pathlib import Path

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
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


# TODO(deepyaman): Leverage `importer` (not supported with v1 compiler).
# https://github.com/kubeflow/pipelines/blob/master/samples/v2/pipeline_with_importer.py
web_downloader_op = kfp.components.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/contrib/web/Download/component-sdk-v2.yaml')


# Define a pipeline and create a task from a component:
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
  # The outputs of the prep_task can be referenced using the
  # prep_task.outputs dictionary: prep_task.outputs['merged_data']


if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
        pipeline_func=nyc_taxi_data_regression_pipeline,
        package_path=str(Path(__file__).parent / 'pipeline.yaml'))
