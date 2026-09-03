import wandb
api = wandb.Api()
artifact = api.artifact("nguyenminhloc2905-bach-khoa-university/MeTRA_Slot_NCO/model-mg9s1sby:v8")
artifact_dir = artifact.download()