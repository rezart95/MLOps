import argparse
import logging
import sys

from azureml.core import Run, Dataset, Datastore, Model
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
    parser.add_argument("--model_path", type=str, help="Model directory")    
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()    
    # current run is the registration step    
    current_run = Run.get_context()    
    ws = current_run.experiment.workspace    
    
    # parent run is the overall pipeline    
    parent_run = current_run.parent    
    logger.info("Parent run id: %s", parent_run.id)    
    
    pipeline_run = PipelineRun(parent_run.experiment, parent_run.id)    
    
    training_run = pipeline_run.find_step_run("train-model")[0]    
    logger.info("Training run: %s", training_run)    
    
    prepare_run = pipeline_run.find_step_run("describe-data")[0]    
    logger.info("Prepare run: %s", prepare_run)    
    
    # Retrieve input training data reference    
    # and register data reference as a dataset    
    datasets = []    
    details = prepare_run.get_details()    
    for name, d in details["runDefinition"]["dataReferences"].items():        
        datastore_name = d["dataStoreName"]        
        if name == "output_dataset":            
            # we only care about the ads_data reference            
            continue        
        logger.info("Dataset info: %s", d)        
        
        path_on_datastore = d["pathOnDataStore"]        
        
        datastore = Datastore.get(ws, datastore_name)        
        dataset = Dataset.Tabular.from_delimited_files(            
            path=(datastore, path_on_datastore)        
        )        
        dataset.register(ws, name, create_new_version=True)        
        datasets.append([name, dataset])    
        
        # Register model    
        model = training_run.register_model(        
            model_name=args.model_name,        
            model_path=args.model_path,        
            tags={"type": "optimization", "area": "ad spend optimization"},        
            model_framework=Model.Framework.CUSTOM,        
            description="Linear programming model to optimize ad spend",        
            datasets=datasets,    
        )
