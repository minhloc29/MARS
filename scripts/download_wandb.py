import wandb
api = wandb.Api()
artifact = api.artifact("nguyenminhloc2905-bach-khoa-university/MeTRA_Slot_NCO/model-5n3h3ne3:v0")
artifact_dir = artifact.download()