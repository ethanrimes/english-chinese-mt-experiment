# Azure setup — step by step

This walkthrough takes you from "Azure account exists" to "first training run
launched". Everything sensitive stays on your machine; what you share with me
(or with anyone collaborating on the repo) is just the *names* of resources,
not their access keys.

> **About sharing credentials with the AI assistant.** You do **not** need to
> paste any API keys, account keys, or service-principal secrets into chat.
> Everything secret lives in your local `.env` file (gitignored) or in Azure
> Key Vault. What I need from you to wire things up is *only* non-secret
> identifiers: subscription ID, resource group, workspace name, region, and
> AML data-asset URIs. Those are listed at the bottom under
> "What to share back".

---

## 0. Prerequisites

- An Azure subscription with **Contributor** (or **Owner**) at minimum on the
  target resource group. If you're using a Microsoft-internal subscription,
  use the one allocated for your team — not your personal sub — and confirm
  the cost center / business justification before provisioning GPU.
- **Azure CLI** (`az`) installed locally:
  - Windows: `winget install Microsoft.AzureCLI`
  - macOS:   `brew install azure-cli`
- The **ML extension** for `az`:
  ```powershell
  az extension add -n ml
  az extension update -n ml
  ```
- The repo cloned locally and the venv set up (you've done this — `.venv/`
  exists and the data splits live under `data/splits/`).

---

## 1. Sign in and pick your subscription

```powershell
az login                                  # opens a browser
az account list -o table                  # see what you have access to
az account set --subscription "<SUBSCRIPTION_ID>"
az account show -o table                  # confirm
```

Copy the **SubscriptionId** into a temporary scratchpad — you'll paste it
into `.env` in step 9.

---

## 2. Request GPU quota (do this first — can take hours)

Almost every new subscription has **zero** GPU quota for any region. You must
request before you can create the compute cluster in step 5.

In the Azure portal:

1. **Subscriptions** → your subscription → **Usage + quotas** (left nav).
2. Filter by **Provider: Compute**.
3. Search for the SKU family you want:
   - For our default cluster (A10): **`Standard NVadsA10v5 Family vCPUs`**
   - For the 5M-scale run (A100): **`Standard NCadsA100v4 Family vCPUs`**
4. Click the pencil icon, set the new limit:
   - A10: request **144 vCPUs** (4 nodes × 36 vCPUs per `NV36ads_A10_v5`).
   - A100: request **96 vCPUs** (4 nodes × 24 vCPUs per `NC24ads_A100_v4`).
5. Submit with a one-line justification ("Research SFT of 360M LM, English↔Chinese MT scaling study").

Most A10 requests come back within 1–4 hours in `eastus` / `westus3`. A100
quota can take 24–48 h and often gets denied in the most-popular regions —
if denied, try `eastus2`, `westus3`, `northeurope`, or `southcentralus`.

**Do not proceed past step 5 until quota is approved**, or `az ml compute
create` will fail with a misleading error.

---

## 3. Create a resource group

```powershell
az group create `
  --name ecmt-rg `
  --location eastus            # match the region you got GPU quota in
```

If you're using a Microsoft-internal subscription that requires tags (cost
center, data classification, etc.), add `--tags Owner=ethankallett ...` as
your team's policy requires — `az group create` will reject the call
otherwise.

---

## 4. Create the AML workspace (auto-provisions storage, key vault, ACR, App Insights)

The repo ships an `azure/workspace.yaml` already; you only need to align the
name/region with what you picked.

```powershell
# In the repo root
az ml workspace create `
  --file azure/workspace.yaml `
  --resource-group ecmt-rg
```

This creates **five** resources side by side:

| Resource | Why |
|---|---|
| Azure ML Workspace | The hub — runs jobs, tracks experiments via MLflow |
| Storage account | Default datastore for datasets + checkpoints |
| Key Vault | Where AML stores connection secrets (W&B key, HF token) |
| Container Registry | Stores our training Docker image |
| App Insights | Job log aggregation |

Naming is automatic and ugly (e.g. `ecmtws8741293`). If you want pretty
names, edit `azure/workspace.yaml` first.

---

## 5. Create the GPU compute cluster

```powershell
az ml compute create `
  --file azure/compute.yaml `
  --workspace-name ecmt-ws `
  --resource-group ecmt-rg
```

The default cluster is `Standard_NV36ads_A10_v5` (1× A10 24GB, auto-scales
0→4 nodes). For the 5M-scale run later, create a second cluster:

```powershell
# Edit the file or do it inline
az ml compute create `
  --name gpu-a100-80g `
  --type amlcompute `
  --size Standard_NC24ads_A100_v4 `
  --min-instances 0 `
  --max-instances 1 `
  --workspace-name ecmt-ws `
  --resource-group ecmt-rg
```

Cluster sits at 0 instances → costs nothing while idle. Spins up when a job
is submitted, scales down 10 minutes after the queue is empty.

---

## 6. Build the training environment (Docker image)

```powershell
az ml environment create `
  --file azure/environment.yaml `
  --workspace-name ecmt-ws `
  --resource-group ecmt-rg
```

**Where does the build run?** Not on your laptop, not on the GPU node. The
build runs on Azure Container Registry's managed build pool (CPU-only, ~10
min, ~$0.10). Your machine just uploads the Dockerfile + build context.

```
local  --upload-->  ACR build task  --pushes image-->  your ACR
                                                            │
                       GPU node pulls image at job time <───┘
                       (~1-2 min, once per fresh node)
```

You **do not need Docker installed locally**. `az ml environment create`
handles the whole flow server-side.

You can — and should — run this step right after step 4 (workspace exists)
while GPU quota is still being approved. The environment build doesn't need
a compute cluster.

First build is slow (5–15 min) because the Dockerfile bakes the COMET model
into the image. Subsequent updates reuse cached layers.

Watch progress in the Azure portal under
**Machine Learning Studio → Assets → Environments → ecmt-pytorch**.

---

## 7. Upload data + extended model as AML data assets

The training jobs need read access to `data/splits/` and `models/extended/`.
The simplest route is to register them as versioned data assets on the
default datastore.

**Important sequencing**: the extended model only exists *after* you've run
the tokenizer + model extension steps locally. The minimum to launch a
training job is:

```powershell
# Pre-flight (if not already done locally)
.\.venv\Scripts\Activate.ps1
python scripts/03_train_tokenizer_extension.py        # ~5-15 min on CPU
python scripts/04_extend_model.py                     # downloads SmolLM2-360M, ~2 min

# Now upload as AML data assets
az ml data create `
  --name ecmt-splits --version 1 `
  --path data/splits --type uri_folder `
  --workspace-name ecmt-ws --resource-group ecmt-rg

az ml data create `
  --name ecmt-extended-model --version 1 `
  --path models/extended --type uri_folder `
  --workspace-name ecmt-ws --resource-group ecmt-rg
```

After upload, the AML data URIs look like
`azureml:ecmt-splits:1` and `azureml:ecmt-extended-model:1`. You'll plug
these into `.env` in step 9.

The splits folder is ~1.2 GB (mostly `scale_5000000.parquet` at 902 MB).
First upload takes 5–10 minutes on a residential connection. The extended
model is ~1.5 GB.

---

## 8. (Optional) Set up Weights & Biases tracking

Skip this if you only want MLflow (Azure ML has MLflow built in — no setup).

If you want W&B too:

1. Get your W&B API key from <https://wandb.ai/authorize>.
2. Register it as an AML workspace connection so the training job can read it
   from a managed secret (not from your local environment):

   ```powershell
   az ml connection create `
     --file azure/wandb_connection.yaml `
     --workspace-name ecmt-ws --resource-group ecmt-rg `
     --set credentials.key=<YOUR_WANDB_KEY>
   ```

   The connection YAML is shipped in this repo at
   `azure/wandb_connection.yaml`. The `--set credentials.key=...` flag
   passes the secret through the CLI rather than committing it.

The training job template (`azure/job_template.yaml`) reads this secret as
`${{secrets.WANDB_API_KEY}}` so it never touches disk in plaintext.

---

## 9. Populate the local `.env` file

Copy the template and fill in:

```powershell
Copy-Item .env.example .env
notepad .env
```

The values you need:

```ini
# === non-secret identifiers ===
AZURE_SUBSCRIPTION_ID=00000000-0000-0000-0000-000000000000
AZURE_RESOURCE_GROUP=ecmt-rg
AZURE_ML_WORKSPACE=ecmt-ws
AZURE_LOCATION=eastus

# AML data-asset URIs (output of step 7)
ECMT_SPLITS_URI=azureml:ecmt-splits:1
ECMT_MODEL_URI=azureml:ecmt-extended-model:1

# === local-only secrets (NEVER committed; .env is gitignored) ===
WANDB_API_KEY=...           # if you set up W&B in step 8
WANDB_PROJECT=english-chinese-mt
WANDB_ENTITY=...            # your W&B username or team
HF_TOKEN=                   # only needed if you add a gated dataset
```

Verify `.env` is gitignored:
```powershell
git check-ignore -v .env
# should print: .gitignore:13:.env  .env
```

---

## 10. Submit the first training job

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/submit_aml_job.py --scales 10000
```

This will:
1. Read `.env` to find your subscription/workspace.
2. Print a per-scale ETA table (first run will say "no prior history; pass
   --sps-override or run the smallest scale first").
3. Render `azure/job_template.yaml` with your data URIs.
4. Submit one AML job per requested scale.

Track progress at
`https://ml.azure.com/experiments/id/ecmt-sft-v1?wsid=/subscriptions/<SUB>/resourceGroups/ecmt-rg/providers/Microsoft.MachineLearningServices/workspaces/ecmt-ws`
(or follow the URL printed by `az ml job create`).

After the first run finishes, `models/runs/timings.jsonl` will have a real
steps-per-second measurement, and subsequent calls to `submit_aml_job.py`
will print confident ETAs for the larger scales.

---

## 11. (Optional) Set a cost guardrail

Before launching the 5M-scale run, set a budget so you can't accidentally
burn $500:

In Azure portal: **Cost Management → Budgets → Add** → scope to your
resource group `ecmt-rg`, set monthly limit (e.g. $150 covers the whole
6-scale sweep with margin), add a 50%/80%/100% alert to your email.

---

## What to share back with me

After you finish steps 1–7, paste these non-sensitive values into chat (or
just confirm `.env` is populated correctly). I can then write any
additional configs, troubleshoot job submissions, or extend the framework:

| Value | Sensitivity | Where to find it |
|---|---|---|
| Subscription ID | low (just identifies which sub) | `az account show --query id -o tsv` |
| Resource group name | low | what you used in step 3 |
| Workspace name | low | what you used in step 4 |
| Region | low | what you picked in step 3 |
| `ECMT_SPLITS_URI` | low | output of step 7 |
| `ECMT_MODEL_URI` | low | output of step 7 |
| GPU quota approval status | low | screenshot of portal works |
| First job's URL | low | output of `submit_aml_job.py` |

**Do NOT paste into chat:**
- W&B API key
- HF token
- Azure storage account key
- Service principal client secret
- Any value from `az ml connection show --include-secrets`

These should live only in `.env` (gitignored) or in Azure Key Vault. If you
ever accidentally paste a secret anywhere, rotate it immediately:
`az ml connection update` for W&B, **Settings → API Keys** at huggingface.co
for HF, etc.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `BadRequest: Quota.Current=0` on `compute create` | GPU quota not approved | Wait for step 2 approval; try another region |
| `ResourceGroupNotFound` | Wrong subscription active | `az account set --subscription <ID>` |
| Job stuck in **Queued** for >15 min | Cluster scaling up first time | Normal; image pull + node bring-up is 5–15 min |
| `az ml job create` errors on the YAML | Old `az ml` extension | `az extension update -n ml` |
| Job logs show "no GPU detected" | Wrong VM SKU on cluster | Recreate compute with a `Standard_NV*` or `Standard_NC*` SKU |
| Out-of-memory on A10 (24 GB) | Batch size too large | `--override trainer.per_device_train_batch_size=8` |
| W&B run never appears | Connection secret not set | `az ml connection show --name wandb-key --include-secrets` to check |
| 5M scale ETA looks scary | Confirm before launching | A10 is too slow for 5M; use A100 cluster, set `--compute gpu-a100-80g` |

If you hit something that doesn't match this table, paste the *error message
only* (no secrets) into chat and I'll work it out.
