import argparse
import logging
import sys

from azureml.core import Run
from azureml.pipeline.core import PipelineRun

logging.basicConfig(    
    format="%(asctime)s %(levelname)-8s [%(name)s]: %(message)s",    
    level=logging.INFO,    
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("register_model")


def parse_args():    
    parser = argparse.ArgumentParser()    
    parser.add_argument(        
        "--model_name", type=str, help="Name under which model will be registered"    
    )    
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()    
    # current run is the notify step    
    current_run = Run.get_context()    
    ws = current_run.experiment.workspace    
    
    # parent run is the overall pipeline    
    parent_run = current_run.parent    
    logger.info("Parent run id: %s", parent_run.id)

    pipeline_run = PipelineRun(parent_run.experiment, parent_run.id)

    register_run = pipeline_run.find_step_run('register-model')[0]    
    logger.info("Training run: %s", register_run)    
    
    training_run = pipeline_run.find_step_run("train-model")[0]    
    logger.info("Training run: %s", training_run)    
    
    prepare_run = pipeline_run.find_step_run("describe-data")[0]    
    logger.info("Prepare run: %s", prepare_run)    
    
    link_to_run = pipeline_run.get_portal_url()    
    
    # Here you could provide the link or other metrics to a notification service    
    # to notify that the pipeline has succeeded by reaching this step.    
    
    # ##### YOUR CODE HERE #####    
    logger.info("Pipeline %s succeeded!  Here is the link to view more %s", args.model_name, link_to_run)    
    ##### YOUR CODE HERE #####
