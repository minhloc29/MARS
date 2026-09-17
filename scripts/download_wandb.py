import wandb
api = wandb.Api()
artifact = api.artifact("nguyenminhloc2905-bach-khoa-university/MeTRA_Slot_NCO/model-dorld0we:v6")
artifact_dir = artifact.download()