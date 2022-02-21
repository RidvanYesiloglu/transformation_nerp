from runset_train import parameters
from runset_train import get_input_runsets_crt_opts
import os
from subprocess import call

def main(opts_strs, params_dict):
    slurm_submit_dir = "/scratch/users/ridvan/jobs/"
    code_dir = ""
    job_script_name = os.path.join(slurm_submit_dir, "latest_job_sc.sh")
    if not os.path.exists(slurm_submit_dir):
        try:
            os.makedirs(slurm_submit_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for no, opts in enumerate(opts_strs):
        args = parameters.get_arguments(params_dict, opts_str=opts)
        repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos])
        #print(repr_str)
        
        waitings_folder = os.path.join("/scratch/users/ridvan/jobs/waiting_jobs", repr_str)
        if not os.path.exists(waitings_folder):
            try:
                os.makedirs(waitings_folder)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        
        job_time = 24
        
        run_commands = ["python train.py{}\n".format(opts)]
                
        script = ""
        script += "#!/bin/bash\n"
        script += "#SBATCH --time={}:00:00\n".format(job_time)
        script += "#SBATCH --job-name={}\n".format(repr_str)
        script += "#SBATCH --mail-user=ridvan@stanford.edu\n"
        script += "#SBATCH --mail-type=BEGIN,END\n" # mail on beginning and end
        
        script +=  "#SBATCH --output={}%j.%x.out\n".format(slurm_submit_dir)
        script +=  "#SBATCH --error={}%j.%x.err\n".format(slurm_submit_dir)

        script += "#SBATCH --nodes=1\n"
        script += "#SBATCH --ntasks-per-node=1\n"
        script += "#SBATCH --partition=gpu\n"
        script += "#SBATCH --gpus=1\n"
        #script += "#SBATCH -C GPU_MEM:32GB" if needed
        
        script += "module load python/2.7\n"
        #script += "module load py-numpy/1.20.3_py39\n"
        script += "module load py-pytorch/1.0.0_py27\n"
        #script += "module load py-matplotlib/3.4.2_py39\n"
        script += "module load py-tensorboardx/1.8_py27\n"
        script += "module load cuda/10.0.130\n"
        

        script += "python3 -m runset_train.remove_from_waitings{}\n".format(opts)
        #script += 'nvidia-smi' if needed
        for command in run_commands:
            script += command 
        job_script_file = open(job_script_name, "w+")
        job_script_file.write(script)
        job_script_file.close()
        rc = call("chmod +x {}".format(job_script_name), shell=True)
        rc = call("srun {}".format(job_script_name), shell=True)
if __name__ == "__main__":
    main()
