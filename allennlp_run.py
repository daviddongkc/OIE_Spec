import json
import shutil
from argparse import ArgumentParser
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
import logging
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    import_submodules('spec_tup')

    parser = ArgumentParser()
    parser.add_argument("--config", default='', help="input config json file")
    parser.add_argument("--model", default='', help="model output directory")
    parser.add_argument("--train_data", default='', help="training file")
    parser.add_argument("--eval_data", default='', help="evaluation file")
    parser.add_argument("--epoch", type=int, default=0, help="number of epoches")
    parser.add_argument("--batch", type=int, default=0, help="batch size")
    parser.add_argument("--cuda", type=int, default=0, help="batch size")
    parser.add_argument("--plm", type=str, default='', help="pretrained language model")

    args = parser.parse_args()

    overrideD = dict()
    overrideD['iterator'] = dict()
    overrideD['trainer'] = dict()
    overrideD['model'] = dict()

    overrideD['trainer']["cuda_device"] = args.cuda

    if args.plm != '':
        overrideD['dataset_reader'] = dict()
        overrideD['validation_dataset_reader'] = dict()
        # roberta-base or  bert-base-uncased or electra
        overrideD['dataset_reader']["bert_model_name"] = args.plm
        overrideD['validation_dataset_reader']["bert_model_name"] = args.plm
        overrideD['model']["bert_model"] = args.plm

    if args.train_data != '' and args.eval_data != '':
        overrideD['train_data_path'] = args.train_data
        overrideD['validation_data_path'] = args.eval_data

    # this section is used for debugging
    if args.model == '' or args.config == '':
        # config_file = "config/spec_bert.json"
        config_file = "config/spec_tup.json"
        serialization_dir = "trained_model/test_dep"
        overrideD['trainer']["num_epochs"] = 5
        overrideD['iterator']["batch_size"] = 16
        # overrideD['train_data_path'] = 'data/lsoie_wiki_test_short.json'
        # erase directory, only applicable to debugging
        shutil.rmtree(serialization_dir, ignore_errors=True)

    else:
        serialization_dir = args.model
        config_file = args.config

    # writing predictions to output folders:
    overrideD['model']["tuple_metric"] = dict()
    overrideD['model']["tuple_metric"]["output_path"] = serialization_dir


    if args.epoch > 0:
        overrideD['trainer']["num_epochs"] = args.epoch
    if args.batch > 0:
        overrideD['iterator']["batch_size"] = args.batch


    overrides = json.dumps(overrideD)


    train_model_from_file(parameter_filename=config_file,
                          serialization_dir=serialization_dir,
                          recover=False,
                          overrides=overrides)

