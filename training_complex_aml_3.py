# For this exercise, we will be looking to add 3 new methods to the PromotionModel class. 
# These 3 methods will include save(), load() and _remove_data(). 
# The _remove_data() function strips out training data from the sub-models in preparation for serialization with the save() method. 
# This reduces the size of the saved model object for better save/load performance. 
# The save() method serializes the PromotionModel object so that it can be uploaded and registered with the AzureML service. 
# By registering the model, it can later be used by inference pipelines/endpoints. The load() method helps facilitate loading a saved model.


# Insert the following code snippet at line 45 in the model.py script.

# model.py

                
def _remove_data(self) -> None:
    for model in self.ols_models.values():
        model.remove_data()
    self.run = None

def save(self, filename: str = "output/model.joblib") -> None:
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    self._remove_data()
    joblib.dump(self, filename)

@staticmethod
def load(path) -> PromotionModel:
    return joblib.load(path)
