# step1: Read through control_script.py. Notice that there is a new pipeline step called notification_step. 
# This step executes the scripts/notify/notify.py script, passing in one parameter which is the name of the model.

# Additionally, the notification_step has to be explicitly chained after the register_model_step, because there is no input dependency from the register step.

# Read through notify.py. Note that there are a few places that information about the previous run is gathered. 
# This information can be passed to a notification system to so indicate the successful completion of a run. 
# For simplicity, we are just looking to log some of this information. 
# On line 49, please write out to the logger that the Pipeline has succeeded, along with the name of the model and a link to the run.


# Insert the following code snippet at line 49 in the notify.py file.

# notify.py

                
logger.info("Pipeline %s succeeded!  Here is the link to view more %s", args.model_name, link_to_run)
