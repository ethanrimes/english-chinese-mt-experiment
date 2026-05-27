# Azure ML setup

These specs define the cloud environment for the data-scaling sweep. The
training code itself is hardware-agnostic — these YAML files just describe
*where* it runs.

## One-time setup

```powershell
# Sign in
az login
az account set -s $env:AZURE_SUBSCRIPTION_ID

# Workspace (skip if you already have one)
az ml workspace create -f azure/workspace.yaml

# GPU compute cluster (auto-scales to zero when idle)
az ml compute create -f azure/compute.yaml \
    -w $env:AZURE_ML_WORKSPACE -g $env:AZURE_RESOURCE_GROUP

# Training environment (PyTorch + our requirements.txt)
az ml environment create -f azure/environment.yaml \
    -w $env:AZURE_ML_WORKSPACE -g $env:AZURE_RESOURCE_GROUP

# Key vault references for secrets (optional but recommended)
az ml connection create --file azure/wandb_connection.yaml \
    -w $env:AZURE_ML_WORKSPACE -g $env:AZURE_RESOURCE_GROUP
```

## Submitting training jobs

```powershell
# One scale
python scripts/submit_aml_job.py --scales 10000

# Several
python scripts/submit_aml_job.py --scales 10000,50000,100000

# All scales (will print total ETA and ask for confirmation if > 4 hours)
python scripts/submit_aml_job.py --all
```

## Compute sizing

| VM SKU | GPU | per-hour | recommended for |
|---|---|---|---|
| Standard_NC6s_v3 | 1× V100 16GB | ~$3.06 | smoke tests (10K-100K scales); use `bf16: false fp16: true` in training.yaml |
| Standard_NC24ads_A100_v4 | 1× A100 80GB | ~$3.40 | main study (default recommendation) |
| Standard_ND96amsr_A100_v4 | 8× A100 80GB | ~$32.77 | only if 5M scale's wallclock matters |

Default in `azure/compute.yaml` is `Standard_NC24ads_A100_v4` with `min_instances=0, max_instances=4` so up to 4 scales train in parallel without burning idle compute.
