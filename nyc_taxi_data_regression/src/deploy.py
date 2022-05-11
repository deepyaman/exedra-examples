import logging

import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

import subprocess


def deploy_azure_ml():
    logger.info("Deploying to Azure ML ...")
    bashCommand = (
        "az ml job create --file ../deploy/azureml/pipeline.yml -g azureml-examples"
    )
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    logger.info("... done.")


def deploy_google_ai():
    logger.info("Deploying to Google AI ...")
    logger.info("... done.")
