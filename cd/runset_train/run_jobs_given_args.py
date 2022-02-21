from runset_train import parameters
from runset_train import train, train_for_all

def main(args_list, params_dict):
    for args in args_list:
        repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos], indRunNo=args.indRunNo)
        print(repr_str)
        print('train for all: ', args.train_for_all)
        if args.train_for_all:
            train_for_all.main(args)
        else:
            train.main(args)
if __name__ == "__main__":
    main()
