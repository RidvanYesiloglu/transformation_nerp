import sys
from runset_train import parameters
import os

params_dict = parameters.decode_arguments_dictionary("params_dictionary")
args = parameters.get_arguments(params_dict)
repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos])
waitings_folder = os.path.join("/scratch/groups/gracegao/" + params_dict.proj_name + "/waiting_jobs", repr_str)
os.rmdir(waitings_folder)
