import argparse

# Parser goes through each line in  theconfiguration file and and sets them as new parameters
def get_parser():
    dict = {}
    parser = argparse.ArgumentParser(description='ML Framework for RNA Seq Expression Data. Check readme for configuration file description.')
    parser.add_argument('-f', '-filename', default='parameters.cfg', help = "Configuration File")
    args = parser.parse_args()
    filename = args.f
    with open(filename) as file:
        for line in file:
              key, value = line.split(' = ')
              dict[key] = value.rstrip('\n')
    return dict
