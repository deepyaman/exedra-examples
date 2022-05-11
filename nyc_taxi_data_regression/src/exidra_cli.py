import click

from deploy import deploy_azure_ml, deploy_google_ai

AZURE_ML_NAME = "AzureML"
GOOGLE_AI_NAME = "GoogleAI"


@click.group()
def cli():
    pass


@cli.command("deploy")
@click.option(
    "--platform",
    required=True,
    type=click.Choice([AZURE_ML_NAME, GOOGLE_AI_NAME]),
    help="Which platform to target.",
)
def deploy(platform):
    """Main entrypoint for exidra deploy."""
    if platform == AZURE_ML_NAME:
        deploy_azure_ml()
    elif platform == GOOGLE_AI_NAME:
        deploy_google_ai()


if __name__ == "__main__":
    cli()
