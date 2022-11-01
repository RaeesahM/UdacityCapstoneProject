import logging
import json
import numpy as np
import joblib
import  os

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from azureml.core.run import Run


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

input_sample = np.array([[6,0.24,'Very Good','J','VVS2',62.8,57,3.94,3.96,2.48]])
output_sample = np.array([5010])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(raw_data):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    logging.info("model 1: request received")
    try:
        result = model.predict(data)
        # You can return any JSON-serializable object.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error

#def run(raw_data):
   # data = json.loads(raw_data)["data"]
   # data = numpy.array(data)
    #result = model.predict(data)
   # logging.info("Request processed")
   # return result.tolist()