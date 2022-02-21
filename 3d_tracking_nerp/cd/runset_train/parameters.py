import sys
#sys.path.append('gnss_code_design/neural_networks')
import argparse
from runset_train import type_check
import itertools
import math
class Param_Info():
    def __init__(self, name, desc, typ, poss, defa, req, ask, cart, shrt_repr):
        self.name = name
        self.desc = desc
        self.typ = typ
        self.poss = [eval(typ)(elem) for elem in poss] if (poss is not None) else None
        self.defa = eval(typ)(defa) if (defa is not None) else None
        self.req = req
        self.ask = ask
        self.cart = int(cart)
        self.shrt_repr = int(shrt_repr)
        
    def const_quest_prompt(self):
        ask_str = self.desc
        if self.poss is not None: ask_str += " (possible values: {})".format(self.poss_vals_to_print())
        if self.defa is not None: ask_str += " (press enter for default {})".format(self.defa)
        ask_str += ": "
        return ask_str
    
    def poss_vals_to_print(self):
        return ', '.join(["{} ({})".format(no+1, el) for no,el in enumerate(self.poss)])
    
    # 
    def get_input_and_update_runsets(self, cart_prod_runsets, vals_list, params_dict):
        names_list = [info.name for info in params_dict.param_infos]
        inp_list = None
        if self.defa is not None:
            inp_list = [self.poss.index(self.defa)+1] if (self.poss is not None) else [self.defa]
        if (self.ask[0] != '0') and ((self.ask[0] == '1') or \
                                     eval('any([val {} for val in vals_list[names_list.index(self.ask[1])]])'.format(self.ask[2]))):
            ask_str = self.const_quest_prompt()
            inp_list = [int(item) if (self.poss is not None) else eval(self.typ)(item) for item in input(ask_str).split()]
            if len(inp_list) != len(set(inp_list)): 
                raise ValueError('Input contains duplicates, no need to do that.')
            if len(inp_list) == 0:
                if self.defa is not None:
                    inp_list = [self.poss.index(self.defa)+1] if (self.poss is not None) else [self.defa]
                else:
                    if (self.req[0] != '0') and ((self.req[0] == '1') or \
                        eval('any([val {} for val in vals_list[names_list.index(self.req[1])]])'.format(self.req[2]))):
                        raise ValueError('You have''nt provided a value!')
        inp_list = self.conv_to_poss(inp_list)
        if len(cart_prod_runsets) > 0:
            new_prod = []
            for no,runset_wo_new in enumerate(list(cart_prod_runsets)):
                if self.cart == 0:
                    mylist = list(itertools.product([runset_wo_new], inp_list if inp_list is not None else [None]))
                    new_prod += [el[0]+(el[1],) for el in mylist]
                else:
                    new_prod += [runset_wo_new + (inp_list[no],)] if inp_list is not None else [runset_wo_new + (inp_list,)]
                        
        else:
            new_prod = list(itertools.product(inp_list))
        
        # conds check:
        
        violated_conds = [] # as list of strings
        for cond in params_dict.conds:
            checked_prod = []
            for run in new_prod:
                expr_str = ""
                cond_can_be_calc = True
                islist = False
                for no, el in enumerate(cond):
                    if no == 0:
                        expr_str += el
                    else:
                        if len(el) == 0:
                            islist = True
                            continue
                        if islist and (names_list.index(el[:el.find(" ")]) < len(vals_list)):
                            expr_str += " vals_list[names_list.index(\'{}\')]".format(el[:el.find(" ")]) + el[el.find(" "):]
                            islist = False
                        elif (not islist) and (names_list.index(el[:el.find(" ")]) < len(run)):
                            expr_str += " run[names_list.index(\'{}\')]".format(el[:el.find(" ")]) + el[el.find(" "):]
                            #islist = False
                        else:
                            cond_can_be_calc = False
                            break
                if cond_can_be_calc and (not eval(expr_str)):
                    if (not (expr_str in violated_conds)):
                        violated_conds.append(expr_str)
                else:
                    checked_prod.append(run)
            new_prod = checked_prod
            if len(new_prod) == 0: raise ValueError("Condition is violated: {} is False.".format(violated_conds))
            vals_list = [[inp for inp in vals if inp in [new_run[no] for new_run in new_prod]] if vals is not None else None for no, vals in enumerate(vals_list)]
        if len(violated_conds) > 0: print("Some cartesian products are removed due to condition(s): {} ".format(violated_conds))
        inp_list = [inp for inp in inp_list if any([inp in [new_run[-1] for new_run in new_prod]])] if inp_list is not None else None
        vals_list.append(inp_list)
        return new_prod, vals_list
    
    def conv_to_poss(self, list_val_nos):
        if self.poss is None:
            return list_val_nos
        noncomplyings = [val_no for val_no in list_val_nos if (val_no>len(self.poss) or val_no<1)]
        if len(noncomplyings)>0:
            raise argparse.ArgumentTypeError("The following is not among possible values: %s. Possible values for %s are %s." \
                                             % (", ".join([str(i) for i in noncomplyings]), self.name, self.poss_vals_to_print()))
        return [self.poss[val_no-1] for val_no in list_val_nos]
class Params_Dict():
    def __init__(self, proj_name, param_infos, conds):
        self.proj_name = proj_name
        self.param_infos = param_infos #list of params_info class instances
        self.conds = conds # list of string lists
def decode_arguments_dictionary(dict_file):
    param_infos = []
    conds = []
    with open(dict_file) as fp:
        for line_no,line in enumerate(fp):
            if line_no == 0: # get the project name
                proj_name = line.strip().split("&")[1].strip()
                continue
            elif line_no == 1: # pass the columns in the dictionary
                continue
            if len(line.strip().split("&")[0].strip().split(";")[0]) == 0:
                conds.append(list(map(str.strip, line.strip().split("&")[1].strip().split(";"))))
                continue
                
            # always length > 0.
            # if length==1 and empty, make it None.
            # if length == 1, convert to list[0] if not possible values.
            # if length > 1 or nonempty possible values, convert to stripped list.
            [name, desc, typ, poss, defa, req, ask, cart, shrt_repr] = [list(map(str.strip, mylist)) if (len(mylist)>1 or (len(mylist[0])>0 and no==3)) \
            else mylist[0] if len(mylist[0])>0 else None for no, mylist in enumerate([text.strip().split(";") for text in line.strip().split("&")])]
            param_infos.append(Param_Info(name, desc, typ, poss, defa, req, ask, cart, shrt_repr))
    params_dict = Params_Dict(proj_name, param_infos, conds)
    return params_dict

# needs the sort in names_list coming from dict and args
def create_repr_str(args, dict_names_list, indRunNo=None, wantShort=False, params_dict=None):
    repr_str = ''
    #dict_names_list = [param_info.name for param_info in params_dict.param_infos]
    #if len(args._get_kwargs()) > len(dict_names_list): raise ValueError('Args somehow includes more than the dictionary.')
    for no, name in enumerate(dict_names_list):
        #print('arg name = ', str(getattr(args, name)) )
        if getattr(args, name) and ( name == 'img_path'):
            print('GIRDIIIIIIIII')
            repr_str += name + '_' + str(getattr(args, name))[-8:-4] + '&'
        elif getattr(args, name) and ((not wantShort) or (params_dict.param_infos[no].shrt_repr==1)):
            repr_str += name + '_' + str(getattr(args, name))+'&'
        
    if indRunNo is not None:
        repr_str += 'indRunNo_' + str(indRunNo)+'&'
    repr_str = repr_str[:-1]
    return repr_str
def check_args(args, params_dict):
    req_check_list = []
    for param_info in params_dict.param_infos:
        if param_info.req[0] == '2': req_check_list.append('(not args.{}) or args.{}'.format(param_info.req[1]+param_info.req[2], param_info.name))
    # req check:
    for req_check in req_check_list:
        if not eval(req_check): raise ValueError("Cond-required param is missing: {} is false.".format(req_check))
    # conds check:
    for cond in params_dict.conds: 
        expr_str = ""
        cond_can_be_calc = True
        islist = False
        for no, el in enumerate(cond):
            if no == 0:
                expr_str += el
            else:
                if len(el) == 0:
                    islist = True
                    continue
                if islist and (el[:el.find(" ")] in [tup[0] for tup in args._get_kwargs()]):
                    expr_str += " [args.{}]".format(el[:el.find(" ")]) + el[el.find(" "):]
                    islist = False
                elif (not islist) and (el[:el.find(" ")] in [tup[0] for tup in args._get_kwargs()]):
                    expr_str += " args.{}".format(el[:el.find(" ")]) + el[el.find(" "):]
                    #islist = False
                else:
                    cond_can_be_calc = False
                    break
        if cond_can_be_calc and (not eval(expr_str)): raise ValueError("Condition is violated: {} is False.".format(expr_str)) 
def get_arguments(params_dict, opts_str=None):
    parser = argparse.ArgumentParser()
    for param_info in params_dict.param_infos:
        parser.add_argument("--"+param_info.name, help=param_info.desc, type=eval(param_info.typ) if not param_info.typ=='type_check.positive_int_tuple' else int, choices=param_info.poss, \
                            default=param_info.defa, required=(param_info.req[0] == '1'), nargs="+" if param_info.typ=='type_check.positive_int_tuple' else 1 )
    if opts_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opts_str.split())
    for param_info in params_dict.param_infos:
        if param_info.typ=='type_check.positive_int_tuple':
            exec("args."+param_info.name+"=tuple(args."+param_info.name+")")
    check_args(args, params_dict)
    return args
