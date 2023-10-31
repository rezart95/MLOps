# Our goal is to write new code that publishes the completed pipeline so that it may be triggered via the SDK, RESTful web API, schedules, and/or events. 
# Once it is published, it defines a shedule for the published pipeline that reacts to events in the datastore. 
# That is, when new files are added or existing files are changed, the pipeline will execute with the path to the changed data passed into the pipeline via a pipeline parameter. 
# Start writing this funcitonality at line 186 of control_script.py


# Insert the following code snippet at line 186 in the control_script.py script.

# control_script.py

                    
                        
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