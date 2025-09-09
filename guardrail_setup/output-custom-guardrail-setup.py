# Databricks notebook source
# MAGIC %md
# MAGIC ## Output Custom Guardrail Example Notebook Using Custom Moderation 

# COMMAND ----------

!pip install mlflow
!pip install openai
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ####0. User-Defined Constants

# COMMAND ----------

# Guardail - custom logic - output
# - Prevent any outputs that are less than or equal to 3 characters in length (some will work, others will not)

# This is the name of the guardrails endpoint
guardrails_endpoint_name = "output-guardrail-min-characters"
# This is where the model will be stored in Unity Catalog (REPLACE CATALOG AND SCHEMA AS NEEDED)
catalogue_path = f"main.default.{guardrails_endpoint_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Define the Custom Guardrail Pyfunc

# COMMAND ----------

# MAGIC %%writefile "./{guardrails_endpoint_name}.py"
# MAGIC
# MAGIC """
# MAGIC To define a custom guardrail pyfunc, the following must be implemented:
# MAGIC 1. def _translate_output_guardrail_request(self, model_input) -> Translates the model input between an OpenAI Chat Completions (ChatV1, https://platform.openai.com/docs/api-reference/chat/create) response and our custom guardrails format.
# MAGIC 2. def invoke_guardrail(self, input) -> Invokes our custom moderation logic.
# MAGIC 3. def _translate_guardrail_response(self, response) -> Translates our custom guardrails response to the OpenAI Chat Completions (ChatV1) format.
# MAGIC 4. def predict(self, context, model_input, params) -> Applies the guardrail to the model input/output and returns the guardrail response.
# MAGIC """
# MAGIC from typing import Any, Dict, List, Union
# MAGIC import json
# MAGIC import copy
# MAGIC import mlflow
# MAGIC from mlflow.models import set_model
# MAGIC import os
# MAGIC import pandas as pd
# MAGIC
# MAGIC class CustomModerationModel(mlflow.pyfunc.PythonModel):
# MAGIC     def __init__(self):
# MAGIC       pass
# MAGIC
# MAGIC     def _invoke_guardrail(self, input: list):
# MAGIC       """ 
# MAGIC       Invokes your guardrail. You may call your APIs here or write custom logic. 
# MAGIC       """
# MAGIC       results = []
# MAGIC
# MAGIC       # Custom guardrail logic that will flag any responses with 3 or fewer characters
# MAGIC       for i, entry in enumerate(input):
# MAGIC         result = {
# MAGIC             "id": i,
# MAGIC             "flagged": False
# MAGIC           }
# MAGIC         if (isinstance(entry, str)):
# MAGIC           if (len(entry) <= 3):
# MAGIC             result["flagged"] = True
# MAGIC         elif (isinstance(entry, dict)):
# MAGIC           if (entry['type'] == "image_url"):
# MAGIC             pass
# MAGIC           elif (entry['type'] == "text"):
# MAGIC             if (len(entry['text']) <= 3):
# MAGIC               result["flagged"] = True
# MAGIC         results.append(result)
# MAGIC
# MAGIC       return results
# MAGIC
# MAGIC     def _translate_output_guardrail_request(self, request: dict):
# MAGIC       """
# MAGIC       Translates a OpenAI Chat Completions (ChatV1) response to our custom guardrail's request format.
# MAGIC       """
# MAGIC       if ("choices" not in request):
# MAGIC         raise Exception(f"Missing key \"choices\" in request: {request}.")
# MAGIC       choices = request["choices"]
# MAGIC       custom_guardrail_input_format = []
# MAGIC
# MAGIC       for choice in choices:
# MAGIC         # Performing validation
# MAGIC         if ("message" not in choice):
# MAGIC           raise Exception(f"Missing key \"message\" in \"choices\": {request}.")
# MAGIC         if ("content" not in choice["message"]):
# MAGIC           raise Exception(f"Missing key \"content\" in \"choices[\"message\"]\": {request}.")
# MAGIC
# MAGIC         custom_guardrail_input_format.append({
# MAGIC           "type": "text",
# MAGIC           "text": choice["message"]["content"]
# MAGIC         })
# MAGIC
# MAGIC       return custom_guardrail_input_format
# MAGIC
# MAGIC     def _translate_guardrail_response(self, response):
# MAGIC       """
# MAGIC       This function translates the custom guardrail's response to the Databricks Guardrails format.
# MAGIC       """
# MAGIC       flagged = any([result["flagged"] for result in response])
# MAGIC       if flagged:
# MAGIC         return {
# MAGIC           "decision": "reject",
# MAGIC           "reject_reason": "Greeting is too short! Must be more than 3 characters long."
# MAGIC         }
# MAGIC       else:
# MAGIC         return {
# MAGIC             "decision": "proceed"
# MAGIC           }
# MAGIC
# MAGIC     def predict(self, context, model_input, params):
# MAGIC         """
# MAGIC         Applies the guardrail to the model input/output and returns a custom guardrail response. 
# MAGIC         """
# MAGIC
# MAGIC         # The input to this model will be converted to a Pandas DataFrame when the model is served
# MAGIC         if (isinstance(model_input, pd.DataFrame)):
# MAGIC             model_input = model_input.to_dict("records")
# MAGIC             model_input = model_input[0]
# MAGIC             assert(isinstance(model_input, dict))
# MAGIC         elif (not isinstance(model_input, dict)):
# MAGIC             return {"decision": "reject", "reject_message": f"Couldn't parse model input: {model_input}"}
# MAGIC           
# MAGIC         try:
# MAGIC           content = self._translate_output_guardrail_request(model_input)
# MAGIC           moderation_response = self._invoke_guardrail(content)
# MAGIC           return self._translate_guardrail_response(moderation_response)
# MAGIC         except Exception as e:
# MAGIC           return {"decision": "reject", "reject_message": f"Errored with the following error message: {e}"}
# MAGIC       
# MAGIC set_model(CustomModerationModel())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Log the custom guardrail to Unity Catalog

# COMMAND ----------

import mlflow
import logging
import json

logging.getLogger("mlflow").setLevel(logging.DEBUG)
model_input_example = {"choices": [
      {
        "index": 0,
        "message": {
            "role": "role",
            "content": "content",
            "refusal": None,
            "annotations": [],
        },
        "logprobs": None,
        "finish_reason": "finish_reason",
      }
    ]
  }


model_path = f"{guardrails_endpoint_name}.py"

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        python_model=model_path,
        name=guardrails_endpoint_name,
        metadata={
          "task":"llm/v1/chat",
        },
        input_example=model_input_example,
        registered_model_name=catalogue_path
    )

# COMMAND ----------

# This cell performs some basic validation for the model, ensuring that the input and output types are being correctly translated.

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

valid_model_input_example = {"choices": [
      {
        "index": 0,
        "message": {
            "role": "role",
            "content": "content",
            "refusal": None,
            "annotations": [],
        },
        "logprobs": None,
        "finish_reason": "finish_reason",
      }
    ]
  }
response = loaded_model.predict(valid_model_input_example)
print(response)
assert("decision" in response)

invalid_model_input_example = {"choices": [
      {
        "index": 0,
        "message": {
            "role": "role",
            "refusal": None,
            "annotations": [],
        },
        "logprobs": None,
        "finish_reason": "finish_reason",
      }
    ]
  }

try:
  loaded_model.predict(invalid_model_input_example)
  assert(False)
except Exception as e:
  print(e)
  assert("Failed to enforce schema of data" in e.message)

fails_guardrail_model_input_example = {"choices": [
      {
        "index": 0,
        "message": {
            "role": "role",
            "content": "hi", # Less than or equal to 3 characters in length; will be rejected/fail guardrail
            "refusal": None,
            "annotations": [],
        },
        "logprobs": None,
        "finish_reason": "finish_reason",
      }
    ]
  }
response = loaded_model.predict(fails_guardrail_model_input_example)
print(response)
assert("decision" in response)
assert(response["decision"] == "reject")

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Create a CPU endpoint for your custom guardrail using the UI
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Attach your custom guardrail to your model serving endpoint using the UI or API (see below)

# COMMAND ----------

import requests
import json

# This is the foundation model endpoint to which you want to attach custom guardrails to
llm_endpoint_name = "" # TODO: Fill this out

# Set your variables
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) + "/"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# Construct the URL
url = f"{host}/api/2.0/serving-endpoints/{llm_endpoint_name}/ai-gateway"

# Define headers
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Define the payload
payload = {
  "guardrails": {
    "output": {
      "custom_guardrails": [
        {
          "endpoint_name": guardrails_endpoint_name
        }
      ]
    }
  }
}

# Make the PUT request
response = requests.put(url, headers=headers, data=json.dumps(payload))

# Print response
print("Status Code:", response.status_code)
print("Response Body:", response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC ####5. Query the serving endpoint (via UI)