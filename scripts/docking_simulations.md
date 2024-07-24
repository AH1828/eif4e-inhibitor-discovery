# Running Docking Simulations with AutoDock Vina on Seawulf Cluster

To run docking simulations with AutoDock Vina on the Seawulf cluster, follow the steps outlined below. This guide covers connecting to the cluster, transferring necessary files, loading required modules, allocating computational resources, and executing docking simulations.

Start by connecting to the Seawulf cluster using SSH. You can connect through the Milan or login servers:

```bash
ssh -X username@milan.seawulf.stonybrook.edu
# or
ssh -X username@login.seawulf.stonybrook.edu
```

Transfer your receptor and ligand structure files from your local machine to the cluster using SCP commands. Ensure you know the path to your files on your local machine and the destination directory on the cluster.

```bash
scp /path/to/local/pure_eIF4E.pdbqt username@milan.seawulf.stonybrook.edu:/gpfs/home/username/destination_directory/
scp -r /path/to/local/potential_compounds username@milan.seawulf.stonybrook.edu:/gpfs/home/username/destination_directory/
```

Once your files are in place, check for available modules and load the necessary ones. You need to load Slurm for resource management and AutoDock Vina for running the docking simulations. Ensure you are using the correct versions of the modules required for your simulations.

```bash
module avail auto  # Check available AutoDock modules
module load slurm  # Load Slurm for resource allocation
module load autodock-vina/1.2.5  # Load AutoDock Vina module
```

Allocate computational resources using Slurm to initiate an interactive bash session on a node with sufficient cores. In this example, a 28-core node is requested. Adjust the number of cores and the partition as needed based on the availability and your specific requirements.

```bash
srun -N 1 -n 28 -p short-28core --pty bash  # Start interactive bash session on a 28-core node
```

Execute the AutoDock Vina command with appropriate parameters, including the paths to your receptor and ligand files, the center coordinates, and the dimensions of the search box. The output will be saved in the specified directory. Always verify the paths to your files and directories to avoid errors.

```bash
vina --receptor /gpfs/home/username/destination_directory/pure_eIF4E.pdbqt \
     --ligand /gpfs/home/username/destination_directory/trial.pdbqt \
     --center_x 0 --center_y 0 --center_z 0 \
     --size_x 20 --size_y 20 --size_z 20 \
     --out /gpfs/home/username/destination_directory/eIF4E_1.pdbqt
```

## Sample Script

Below is a complete sample script summarizing the steps above:

```bash
# Connect to the Seawulf cluster
ssh -X username@milan.seawulf.stonybrook.edu
# or
ssh -X username@login.seawulf.stonybrook.edu

# Transfer necessary files
scp /path/to/local/pure_eIF4E.pdbqt username@milan.seawulf.stonybrook.edu:/gpfs/home/username/destination_directory/
scp -r /path/to/local/potential_compounds username@milan.seawulf.stonybrook.edu:/gpfs/home/username/destination_directory/

# Load required modules
module avail auto
module load slurm
module load autodock-vina/1.2.5

# Allocate computational resources and start an interactive bash session
srun -N 1 -n 28 -p short-28core --pty bash

# Run docking simulations with AutoDock Vina
vina --receptor /gpfs/home/username/destination_directory/pure_eIF4E.pdbqt \
     --ligand /gpfs/home/username/destination_directory/trial.pdbqt \
     --center_x 0 --center_y 0 --center_z 0 \
     --size_x 20 --size_y 20 --size_z 20 \
     --out /gpfs/home/username/destination_directory/eIF4E_1.pdbqt
```
