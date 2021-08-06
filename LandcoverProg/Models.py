import os
import json

import logging
LOGGER = logging.getLogger("server")


def _load_model(model):
    if "fn" in model["model"]:
        if not os.path.exists(model["model"]["fn"]):
            return False
    return model["model"]

def load_models():
    models = dict()
    
    model_json = json.load(open("models.json", "r"))
    for key, model in model_json.items():
        model_object = _load_model(model)
        
        if model_object is False:
            LOGGER.error("Model files are missing, we will not be able to serve the following model: '%s'" % (key)) 
        else:
            models[key] = model_object

    return models