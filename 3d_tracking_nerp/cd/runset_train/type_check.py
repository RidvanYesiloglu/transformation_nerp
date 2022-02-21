import argparse
def positive_integer(value):
    cnv_value = int(value)
    if cnv_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive integer value" % value)
    return cnv_value
def nonnegative_integer(value):
    cnv_value = int(value)
    if cnv_value < 0:
        raise argparse.ArgumentTypeError("%s is an invalid nonnegative integer value" % value)
    return cnv_value
def positive_float(value):
    cnv_value = float(value)
    if cnv_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return cnv_value
def dictionary(value):
    cnv_value = dict(eval(value))
    return cnv_value
def positive_int_tuple(value):
    cnv_value = tuple(eval(value))
    return cnv_value
def boolean(value):
    cnv_value = bool(int(value))
    return cnv_value