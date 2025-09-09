# mosaic-ai-gateway-inference-management
Leverage Mosaic AI Gateway's custom guardrail, rate limiting, and fallback features for better model inference management!

This repository shows how to use [Mosaic AI Gateway's](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints):
- User & Group-Based Rate Limits
- Bring-Your-Own Custom Guardrails
- Live Compare/Evaluate Multiple Models
- Model Fallback

## Guardrail Setup
Run these first (with modifications as needed) and add guardrails to your custom endpoint.
- [Input Guardrails](https://github.com/mohamad-aboufoul-dbx/mosaic-ai-gateway-inference-management/blob/main/guardrail_setup/input-custom-guardrail-setup.py)
- [Output Guardrails](https://github.com/mohamad-aboufoul-dbx/mosaic-ai-gateway-inference-management/blob/main/guardrail_setup/output-custom-guardrail-setup.py)

## Mosaic AI Gateway Notebook
Use the [inference management notebook](https://github.com/mohamad-aboufoul-dbx/mosaic-ai-gateway-inference-management/blob/main/mosaic-ai-gateway-inference-management-notebook.py) to try out the Mosaic AI Gateway features.
- Follow instructions in the notebook to make adjustments as needed.
