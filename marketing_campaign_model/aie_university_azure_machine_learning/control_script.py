#!/usr/bin/env python

# Imports
from azureml.core import Environment, Experiment, RunConfiguration, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.datastore import Datastore
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.pipeline.core import (
    Pipeline,
    PipelineData,
    PipelineParameter,
    PublishedPipeline,
)
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core.schedule import Schedule


#############################
# Prepare the AML workspace #
#############################

# The tenant ID can be viewed in the Azure portal.
# Click the "Directories + subscriptions" button in the top bar. It looks like a book with a funnel.
# Copy-paste your directory ID into the tenant_id argument below.
interactive_auth = InteractiveLoginAuthentication(
    tenant_id="69c4b823-534e-4301-a2ff-44cac721ac0f"
)

# Download the workspace config.json from the AML portal and save it beside this script.
ws = Workspace.from_config(path="config.json", auth=interactive_auth)

##########################
# Register training data #
##########################
default_datastore = ws.get_default_datastore()
keyvault = ws.get_default_keyvault()


# # Register the blob container as a Datastore
training_datastore = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name="ads_data",
    container_name="content",
    account_name=default_datastore.account_name,
    account_key=keyvault.get_secret("storage-key"),
)


############################
# Create a compute cluster #
############################

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    # To use a different region for the compute, add a location='<region>' parameter
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2",
        max_nodes=4,
        idle_seconds_before_scaledown=900,
        tags={"Project": "AIEU"},
    )
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    cpu_cluster.wait_for_completion(show_output=True)


#################
# Add job steps #
#################

# --- Step 1: describe-data step ---

# Use a curated environment for the describe-data script to run on a compute cluster
# See https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments
# for more info and a list of available curated environments
step1_environment = Environment.get(
    ws, name="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu", version=28
)

# Define a run configuration with the environment and compute target
step1_run_config = RunConfiguration()
step1_run_config.docker.use_docker = True
step1_run_config.environment = step1_environment
step1_run_config.target = ws.compute_targets[cpu_cluster_name]

output_dataset = PipelineData("output_dataset")
input_datapath = DataPath(
    datastore=training_datastore,
    path_on_datastore="ads_data_20161101.csv",
    name="ads_data",
)
input_path_parameter = PipelineParameter(
    name="input_data", default_value=input_datapath
)
input_path = (input_path_parameter, DataPathComputeBinding(mode="mount"))

describe_data_step = PythonScriptStep(
    name="describe-data",
    source_directory="scripts/describe_dataset",
    script_name="describe_dataset.py",
    arguments=["--input-path", input_path, "--output-dataset", output_dataset],
    inputs=[input_path],
    outputs=[output_dataset],
    runconfig=step1_run_config,
    allow_reuse=False,
)

# --- Step 2: model training step ---
# Use a custom environment for the train-model script to run on a compute cluster
try:
    step2_environment = Environment.get(ws, name="train-model")
except Exception:
    step2_environment = Environment.from_conda_specification(
        name="train-model", file_path="scripts/model/environment.yaml"
    )
    step2_environment.register(ws)

# Define a run configuration with the environment and compute target
step2_run_config = RunConfiguration()
step2_run_config.docker.use_docker = True
step2_run_config.environment = step2_environment
step2_run_config.target = ws.compute_targets[cpu_cluster_name]

# define pipeline parameter for setting seed in model script
seed_parameter = PipelineParameter(name="input_data", default_value=755)

model_train_step = PythonScriptStep(
    name="train-model",
    source_directory="scripts/model/",
    script_name="model.py",
    arguments=["--input-dataset", output_dataset],
    inputs=[output_dataset],
    runconfig=step2_run_config,
    allow_reuse=False,
)

# --- Step 3: Register the trained model ---
# Use a custom environment for the train-model script to run on a compute cluster
try:
    step3_environment = Environment.get(ws, name="register-model")
except Exception:
    step3_environment = Environment.from_conda_specification(
        name="register-model", file_path="scripts/register_model/environment.yaml"
    )
    step3_environment.register(ws)

step3_run_config = RunConfiguration()
step3_run_config.docker.use_docker = True
step3_run_config.environment = step3_environment
step3_run_config.target = ws.compute_targets[cpu_cluster_name]

register_model_step = PythonScriptStep(
    name="register-model",
    source_directory="scripts/register_model",
    script_name="register.py",
    arguments=[
        "--model_name",
        "promotion-model",
        "--model_path",
        "outputs/model.joblib",
    ],
    runconfig=step3_run_config,
    allow_reuse=False,
)

# Required, as there is no implicit data dependency between the train and register steps
register_model_step.run_after(model_train_step)

# --- Step 4: Notify that the pipeline is complete ---

try:
    step4_environment = Environment.get(ws, name="notify")
except Exception:
    step4_environment = Environment.from_conda_specification(
        name="notify", file_path="scripts/notify/environment.yaml"
    )
    step4_environment.register(ws)

step4_run_config = RunConfiguration()
step4_run_config.docker.use_docker = True
step4_run_config.environment = step4_environment
step4_run_config.target = ws.compute_targets[cpu_cluster_name]

notification_step = PythonScriptStep(
    name="notify",
    source_directory="scripts/notify",
    script_name="notify.py",
    arguments=[
        "--model_name",
        "promotion-model",
    ],
    runconfig=step4_run_config,
    allow_reuse=False,
)

# Required, as there is no implicit data dependency between the notify and register steps
notification_step.run_after(register_model_step)

# construct the pipeline
pipeline_steps = [describe_data_step, model_train_step, register_model_step, notification_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

# # Check for errors in the pipeline.
pipeline_errors = pipeline.validate()
assert (
    len(pipeline_errors) == 0
), f"There are errors in the pipeline.\n{pipeline_errors}"

#####################################
# Publish and schedule the pipeline #
#####################################
pipeline_name = "aieu-training-pipeline"
pipeline_version = "1.0"
published_pipeline = None

# Check for an existing published pipeline so that we don't duplicate it.
for pp in PublishedPipeline.list(ws):
    if pp.name == pipeline_name and pp.version == pipeline_version:
        published_pipeline = pp

if published_pipeline is None:
    published_pipeline = pipeline.publish(
        name=pipeline_name,
        description="Published AIEU Model Training Pipeline",
        version="1.0",
    )

# Check for an existing schedule so we don't duplicate it.
schedules = Schedule.get_schedules_for_pipeline_id(ws, published_pipeline.id)
if not schedules:
    reactive_schedule = Schedule.create(
        ws,
        name="ReactiveToDataSchedule",
        description="Trigger pipeline based on input file change.",
        pipeline_id=published_pipeline.id,
        datastore=training_datastore,
        experiment_name="training-pipeline",
        data_path_parameter_name="input_data",
    )

# submit the pipeline
experiment = Experiment(ws, "training-pipeline")
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)
