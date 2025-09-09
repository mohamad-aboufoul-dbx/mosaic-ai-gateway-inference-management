# Databricks notebook source
# MAGIC %md
# MAGIC ## Input Custom Guardrail Example Notebook Using Custom Moderation 

# COMMAND ----------

!pip install mlflow
!pip install openai
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ####0. User-Defined Constants

# COMMAND ----------

import os

# Guardrail - LLM - input
# - Prevent any inputs that contain a greeting (basic prompt)

# This is the name of the guardrails endpoint
guardrails_endpoint_name = "input-guardrail-llm-no-greetings"
# This is where the model will be stored in Unity Catalog (REPLACE CATALOG AND SCHEMA AS NEEDED)
catalogue_path = f"main.default.{guardrails_endpoint_name}"

password = dbutils.secrets.get(scope = "maboufoul_ai_gateway_testing", key = "DATABRICKS_TOKEN")

os.environ["DATABRICKS_TOKEN"] = password

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Define the Custom Guardrail Pyfunc

# COMMAND ----------

# MAGIC %%writefile "./{guardrails_endpoint_name}.py"
# MAGIC
# MAGIC """
# MAGIC To define a custom guardrail pyfunc, the following must be implemented:
# MAGIC 1. def _translate_input_guardrail_request(self, model_input) -> Translates the model input between an OpenAI Chat Completions (ChatV1, https://platform.openai.com/docs/api-reference/chat/create) request and our custom guardrails format.
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
# MAGIC from openai import OpenAI
# MAGIC
# MAGIC class CustomModerationModel(mlflow.pyfunc.PythonModel):
# MAGIC     def __init__(self):
# MAGIC       self.client = OpenAI(
# MAGIC         api_key=os.environ.get("DATABRICKS_TOKEN"),
# MAGIC         base_url="https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints"
# MAGIC       )
# MAGIC       self.model_name = "databricks-meta-llama-3-3-70b-instruct"  # Pay-per-token model for input guardrail
# MAGIC
# MAGIC     def _invoke_guardrail(self, input: list):
# MAGIC       """ 
# MAGIC       Invokes your guardrail. You may call your APIs here or write custom logic. 
# MAGIC       """
# MAGIC       results = []
# MAGIC
# MAGIC       # Custom guardrail logic that will flag any greetings (using an LLM)
# MAGIC       for i, entry in enumerate(input):
# MAGIC         result = {
# MAGIC             "id": i,
# MAGIC             "flagged": False
# MAGIC           }
# MAGIC         if (isinstance(entry, str)):
# MAGIC           chat_completion = self.client.chat.completions.create(
# MAGIC             messages=[
# MAGIC             {
# MAGIC               "role": "system",
# MAGIC               "content": """Respond with "GREETING" if the input contains a greeting and "NO-GREETING" otherwise. Respond with NOTHING ELSE!"""
# MAGIC             },
# MAGIC             {
# MAGIC               "role": "user",
# MAGIC               "content": entry
# MAGIC             }
# MAGIC             ],
# MAGIC             model="databricks-meta-llama-3-1-8b-instruct", 
# MAGIC             max_tokens=10
# MAGIC           )
# MAGIC           entry_type = chat_completion.choices[0].message.content
# MAGIC
# MAGIC           if (entry_type == "GREETING"):
# MAGIC             result["flagged"] = True
# MAGIC           # if ("tax" in entry):
# MAGIC           #   result["flagged"] = True
# MAGIC         elif (isinstance(entry, dict)):
# MAGIC           if (entry['type'] == "image_url"):
# MAGIC             pass
# MAGIC           elif (entry['type'] == "text"):
# MAGIC             chat_completion = self.client.chat.completions.create(
# MAGIC               messages=[
# MAGIC               {
# MAGIC                 "role": "system",
# MAGIC                 "content": """Respond with "GREETING" if the input contains a greeting and "NO-GREETING" otherwise. Respond with NOTHING ELSE!"""
# MAGIC               },
# MAGIC               {
# MAGIC                 "role": "user",
# MAGIC                 "content": entry['text']
# MAGIC               }
# MAGIC               ],
# MAGIC               model="databricks-meta-llama-3-1-8b-instruct", 
# MAGIC               max_tokens=10
# MAGIC             )
# MAGIC             entry_type = chat_completion.choices[0].message.content
# MAGIC
# MAGIC             if (entry_type == "GREETING"):
# MAGIC               result["flagged"] = True
# MAGIC             # if ("tax" in entry['text']):
# MAGIC             #   result["flagged"] = True
# MAGIC         results.append(result)
# MAGIC
# MAGIC       return results
# MAGIC     
# MAGIC     def _translate_input_guardrail_request(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
# MAGIC       """
# MAGIC       Translates an OpenAI Chat Completions (ChatV1) request to our custom guardrail's request format. 
# MAGIC       """
# MAGIC       if ("messages" not in request):
# MAGIC         raise Exception("Missing key \"messages\" in request: {request}.")
# MAGIC       messages = request["messages"]
# MAGIC       custom_guardrail_input_format = []
# MAGIC
# MAGIC       for message in messages: 
# MAGIC         # Performing validation
# MAGIC         if ("content" not in message):
# MAGIC           raise Exception("Missing key \"content\" in \"messages\": {request}.")
# MAGIC
# MAGIC         content = message["content"]
# MAGIC         if (isinstance(content, str)):
# MAGIC           custom_guardrail_input_format.append({"type": "text", "text": content})
# MAGIC         elif (isinstance(content, list)):
# MAGIC           for item in content:
# MAGIC             if (item["type"] == "text"):
# MAGIC               custom_guardrail_input_format.append({"type": "text", "text": item["text"]})
# MAGIC             elif (item["type"] == "image_url"):
# MAGIC               custom_guardrail_input_format.append({"type": "image_url", "image_url": {"url": item["image_url"]["url"]}})
# MAGIC         else:
# MAGIC           raise Exception(f"Invalid value type for \"content\": {request}")
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
# MAGIC           "reject_reason": "Sorry, the LLM is not accepting greetings at this time!"
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
# MAGIC         # The input to this model will be converted to a Pandas DataFrame when the model is served
# MAGIC         if (isinstance(model_input, pd.DataFrame)):
# MAGIC             model_input = model_input.to_dict("records")
# MAGIC             model_input = model_input[0]
# MAGIC             assert(isinstance(model_input, dict))
# MAGIC         elif (not isinstance(model_input, dict)):
# MAGIC             return {"decision": "reject", "reject_message": f"Couldn't parse model input: {model_input}"}
# MAGIC           
# MAGIC
# MAGIC         try:
# MAGIC           content = self._translate_input_guardrail_request(model_input)
# MAGIC           moderation_response = self._invoke_guardrail(content)
# MAGIC           return self._translate_guardrail_response(moderation_response)
# MAGIC         except Exception as e:
# MAGIC           return {"decision": "reject", "reject_message": f"Errored with the following error message: {e}"}
# MAGIC       
# MAGIC set_model(CustomModerationModel())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Log the custom guardrail to Unity Catalogue

# COMMAND ----------

import mlflow
import logging

logging.getLogger("mlflow").setLevel(logging.DEBUG)
model_input_example = {
    "messages": [{"role": "role", "content": "content"}]
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

valid_model_input_example = {
    "messages": [{"role": "user", "content": "content"}]
  }
response = loaded_model.predict(valid_model_input_example)
print(response)
assert("decision" in response)

invalid_model_input_example = {
    "messages": [{"role": "role"}]
  }

try:
  loaded_model.predict(invalid_model_input_example)
  assert(False)
except Exception as e:
  print(e)
  assert("Failed to enforce schema of data" in e.message)

fails_guardrail_model_input_example = {
    "messages": [{"role": "role", "content": "hello!"}]
  }
response = loaded_model.predict(fails_guardrail_model_input_example)
print(response)
assert("decision" in response)
assert(response["decision"] == "reject")

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Create a CPU endpoint for your custom guardrail using the UI

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
    "input": {
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