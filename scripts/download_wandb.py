import wandb
api = wandb.Api()
artifact = api.artifact("nguyenminhloc2905-bach-khoa-university/MeTRA_Slot_NCO/model-u3cp2s4t:v0")
artifact_dir = artifact.download()