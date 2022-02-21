from runset_train import parameters
import sys
def main():
    dict_file = "../params_dictionary"
    params_dict = parameters.decode_arguments_dictionary(dict_file)
    args = parameters.get_arguments(params_dict)
    repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos])
    sys.stdout.write(repr_str)
    sys.exit(0)

if __name__ == "__main__":
    main()
