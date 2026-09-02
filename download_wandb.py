import wandb
api = wandb.Api()
artifact = api.artifact("nguyenminhloc2905-bach-khoa-university/MeTRA_Slot_NCO/model-hf0yxtb6:v13")
artifact_dir = artifact.download()