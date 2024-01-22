import wandb

entity = 'w-chunyang'
project_name = 'warpmesh'

dataset_path = "/Users/chunyang/projects/WarpMesh/data/dataset.tar.gz"

run = wandb.init(project=project_name, job_type="add-dataset")
artifact = wandb.Artifact(name="dataset", type="dataset")
artifact.add_dir(local_path=dataset_path)
run.log_artifact(artifact)