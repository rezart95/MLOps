# In control_script.py, we want to create a new pipeline step called model_train_step. 
# This step will execute the scripts/model/model.py script, passing in two parameters, one of which is the output data from the describe-data step. 
# The other parameter will be a seed used to control model reproducibility.
# Additionally, the model_train_step will need to be able to specify the need to use a custom environment that is built from a conda environment 
# file located alonside model.py called environment.yaml. This new pipeline step should be created at line 127 of control_script.py.



# Insert the following code snippet at line 127 in the control_script.py file.

# control_script.py

                
model_train_step = PythonScriptStep(
    name="train-model",
    source_directory="scripts/model/",
    script_name="model.py",
    arguments=["--input-dataset", output_dataset],
    inputs=[output_dataset],
    runconfig=step2_run_config,
    allow_reuse=False,
)
                