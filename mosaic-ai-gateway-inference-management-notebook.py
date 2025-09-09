# Databricks notebook source
# MAGIC %md
# MAGIC ## Before Running
# MAGIC - Make sure you have your input & output guardrails established (see setup notebooks for how to do so)
# MAGIC   - You can modify the code in each as needed!

# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

# MAGIC %md
# MAGIC ## User & Group-Based Rate Limits (GA)

# COMMAND ----------

from openai import OpenAI
import os


DATABRICKS_TOKEN = dbutils.secrets.get(scope = "maboufoul_ai_gateway_testing", key = "DATABRICKS_TOKEN") # REPLACE WITH REFERENCE TO YOUR OWN ACCESS TOKEN (https://docs.databricks.com/aws/en/dev-tools/auth/)

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints"
)

llm_endpoint_name = "llama-v3_1-8b"

for i in range(5):
  try:
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "Respond with a short greeting (2 words max)."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
      ],
      model=llm_endpoint_name, 
      max_tokens=20
    )

    print(chat_completion.choices[0].message.content)
  except Exception as e:
    print(f"ERROR: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC https://docs.databricks.com/api/workspace/servingendpoints/putaigateway#rate_limits

# COMMAND ----------

# DBTITLE 1,Update Rate Limiting by User Programmatically - CHANGE CODE
import requests
import json

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

payload = {
  "rate_limits": [
    {
      "calls": 2, # Limiting username to 2 queries per minute
      "key": "user", # Currently, 'user', 'user_group, 'service_principal', and 'endpoint' are supported, with 'endpoint' being the default if not specified.
      "principal": "mohamad.aboufoul@databricks.com",
      "renewal_period": "minute" # Currently, only 'minute' is supported. (QPM)
    }
  ]
}

# Make the PUT request
response = requests.put(url, headers=headers, data=json.dumps(payload))

# Print response
print("Status Code:", response.status_code)
print("Response Body:", response.json())

# COMMAND ----------

for i in range(5):
  try:
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "Respond with a short greeting (2 words max)."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
      ],
      model=llm_endpoint_name, 
      max_tokens=20
    )

    print(chat_completion.choices[0].message.content)
  except Exception as e:
    print(f"ERROR: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bring Your Own Custom Guardrails

# COMMAND ----------

input_guardrails_endpoint_name = "ai-gateway-input-guardrail-llm-no-greetings" # Made via `input-custom-guardrail-setup` notebook (Prevents any inputs that contain a greeting (basic prompt))
output_guardrails_endpoint_name = "ai-gateway-output-guardrail-min-characters" # Made via `output-custom-guardrail-setup` notebook (Prevents any outputs that are less than or equal to 3 characters in length (some will work, others will not))

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
          "endpoint_name": input_guardrails_endpoint_name
        }
      ]
    },
    "output": {
      "custom_guardrails": [
        {
          "endpoint_name": output_guardrails_endpoint_name
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

# Guardrail - LLM - input
# - Prevent any inputs that contain a greeting (basic prompt)
# - Test after applying INPUT guardrail

for i in range(5):
  try:
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "Respond with a short greeting (2 words max)."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
      ],
      model=llm_endpoint_name, 
      max_tokens=20
    )

    print(chat_completion.choices[0].message.content)
  except Exception as e:
    print(f"ERROR: {e}")

# COMMAND ----------

# Guardail - custom logic - output
# - Prevent any outputs that are less than ___ characters long (some will work, others will not)
# - Test after applying OUTPUT guardrail

for i in range(5):
  try:
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "Respond with a short greeting (2 words max)."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
      ],
      model=llm_endpoint_name, 
      max_tokens=20
    )

    print(chat_completion.choices[0].message.content)
  except Exception as e:
    print(f"ERROR: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Live compare/evaluate two models
# MAGIC - Wrap other provisioned throughput endpoints [`llama-v3_1-8b` and `llama-v3-8b`] in custom endpoint (treat them like external models)
# MAGIC   - NOTE: MAKE SURE TO HAVE [INFERENCE TABLES ENABLED](https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables#enable-and-disable-inference-tables) FOR THE ENDPOINT!
# MAGIC - In "served entities" add another and distribute traffic percentage accordingly
# MAGIC - Use the new custom endpoint for the code below

# COMMAND ----------

from openai import OpenAI
import os

# Obtain temporary token via notebook
DATABRICKS_TOKEN = dbutils.secrets.get(scope = "maboufoul_ai_gateway_testing", key = "DATABRICKS_TOKEN")

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints"
)

llm_endpoint_name = "multi_version_llama_8b"

for i in range(100):
  try:
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "Respond with a short greeting (2 words max)."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
      ],
      model=llm_endpoint_name, 
      max_tokens=20
    )

    print(chat_completion.choices[0].message.content)
  except Exception as e:
    print(f"ERROR: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC - Check table [inference table of the custom endpoint] once populated and view distribution
# MAGIC   - Change references and date range as needed below
# MAGIC - If needed, join with inference tables of underlying entities [entities in the custom endpoint - `llama-v3_1-8b` and `llama-v3-8b`]

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM system.serving.served_entities
# MAGIC   WHERE served_entity_id IN ('7467bb7e254b4a79be66784351708f71', 'da606c132ba142539f088f71ae443922')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT served_entity_name, count(*) AS requests FROM main.mohamad_aboufoul.multi_version_llama_8b_payload mvl
# MAGIC   LEFT JOIN (
# MAGIC     SELECT served_entity_id, served_entity_name FROM system.serving.served_entities
# MAGIC   ) se ON mvl.served_entity_id = se.served_entity_id
# MAGIC   WHERE request_time > '2025-07-09T09:14:45.314-07:00' AND request_time < '2025-07-09T09:16:52.988-07:00'
# MAGIC GROUP BY served_entity_name
# MAGIC ORDER BY requests DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fallback from an LLM to a custom model
# MAGIC - Wrap other provisioned throughput endpoints in custom endpoint (treat them like external models)
# MAGIC - Can enable fallbacks on this
# MAGIC - Use the new custom endpoint for the code below as well
# MAGIC - We added rate limit of 10 QPM for the entity `llama-v3_1-8b`
# MAGIC

# COMMAND ----------

for i in range(100):
  try:
    chat_completion = client.chat.completions.create(
      messages=[
      {
        "role": "system",
        "content": "Respond with a short greeting (2 words max)."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
      ],
      model=llm_endpoint_name, 
      max_tokens=20
    )

    print(chat_completion.choices[0].message.content)
  except Exception as e:
    print(f"ERROR: {e}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT served_entity_name, count(*) AS requests FROM main.mohamad_aboufoul.multi_version_llama_8b_payload mvl
# MAGIC   LEFT JOIN (
# MAGIC     SELECT served_entity_id, served_entity_name FROM system.serving.served_entities
# MAGIC   ) se ON mvl.served_entity_id = se.served_entity_id
# MAGIC   WHERE request_time > '2025-07-09T09:19:42.558-07:00' AND request_time < '2025-07-09T09:21:52.609-07:00'
# MAGIC GROUP BY served_entity_name
# MAGIC ORDER BY requests DESC