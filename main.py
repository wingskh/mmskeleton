#!/usr/bin/env python
import argparse
import sys
import yaml
import torch

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    # processors['recognition'] = import_class('mmskeleton.processor.recognition.REC_Processor')
    # processors['demo_old'] = import_class('mmskeleton.processor.demo_old.Demo')
    # processors['demo'] = import_class('mmskeleton.processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('mmskeleton.processor.demo_offline.DemoOffline')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    with open(arg.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)

    raw_weight_path = str(default_arg['weights'])
    raw_weight = torch.load(raw_weight_path)

    if set(raw_weight.keys()) == set({'meta', 'optimizer', 'state_dict'}): # convention of new model
        converted_path = raw_weight_path.rsplit('.', maxsplit=1)[0] + '.pt'
        torch.save(raw_weight['state_dict'], converted_path)   # save the pt version of the model
        default_arg['weights'] = str(converted_path)

    with open(arg.config, "w") as f:
        yaml.dump(default_arg, f)
    # start
    Processor = processors[arg.processor]
    print(sys.argv[2:], "========================", file=sys.stdout)
    p = Processor(sys.argv[2:])
    p.start()