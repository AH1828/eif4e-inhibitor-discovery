# Running Docking Simulations with AutoDock Vina

To run docking simulations with AutoDock Vina on the Seawulf cluster, start by connecting to the cluster using SSH, either through the Milan or login servers. Next, transfer your necessary files, such as receptor and ligand structures, from your local machine to the cluster using SCP commands. Once the files are in place, check for available modules and load the required ones, particularly Slurm and AutoDock Vina. Allocate computational resources using Slurm to initiate an interactive bash session on a node with sufficient cores. Finally, execute the AutoDock Vina command with appropriate parameters, including receptor and ligand file paths, center coordinates, and box dimensions, to perform the docking simulation. The output will be saved in the specified directory, allowing for further analysis. This workflow ensures efficient use of the Seawulf cluster's computational power for molecular docking studies.

## Sample script

```bash
ssh -X username@milan.seawulf.stonybrook.edu # or login.seawulf.stonybrook.edu
scp /.../pure_eIF4E.pdbqt username@milan.seawulf.stonybrook.edu:/gpfs/home/username/.../
scp -r /.../potential_compounds username@milan.seawulf.stonybrook.edu:/gpfs/home/username/.../

module avail auto # check the available modules
module load slurm # resource allocation
module load autodock-vina/1.2.5
srun -N 1 -n 28 -p short-28core --pty bash # start interactive bash session on 28-core node
vina --receptor /gpfs/home/xingma/MyDirectory/pure_eIF4E.pdbqt --ligand /gpfs/home/xingma/MyDirectory/trial.pdbqt --center_x 0 --center_y 0 --center_z 0 --size_x 20 --size_y 20 --size_z 20 --out /gpfs/home/xingma/MyDirectory/eIF4E_1.pdbqt
```
