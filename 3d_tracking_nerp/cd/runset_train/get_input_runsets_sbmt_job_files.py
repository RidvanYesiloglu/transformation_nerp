from runset_train import get_input_runsets_crt_opts
from runset_train import submit_job_files_given_opts
from runset_train import run_jobs_given_args
import os
import torch
def main():
    print('TORCH version is {}'.format(torch.__version__))
    dict_file = "params_dictionary"
    slurmOrLocal, optsOrArgs, params_dict = get_input_runsets_crt_opts.main(dict_file)
    if slurmOrLocal == 1:
        submit_job_files_given_opts.main(optsOrArgs.split("\n"), params_dict)
    else:
        run_jobs_given_args.main(optsOrArgs, params_dict)

if __name__ == "__main__":
    main()
