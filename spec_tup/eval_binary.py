from overrides import overrides
from allennlp.training.metrics.metric import Metric
import os
import json
from collections import defaultdict
import numpy as np
from benchmark.oie_readers.tabReader import TabReader
from benchmark.matcher import Matcher
from benchmark.benchmark import Benchmark
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, accuracy_score
@Metric.register("spec_match_binary")
class Carb(Metric):
    """
    Computes scores according to carb framework
    """
    def __init__(self, output_path: str = None, dev_set: str = None):
        super(Carb, self).__init__()
        self._all_sentences = []
        self._all_tokens = []
        self._all_predictions = []
        self._all_confidences = []
        self._all_predicate_id = []
        self._spec_sent = []
        self._spec_sent_gold = []
        self._spec_sent_pred = []
        self._spec_sent_level = []
        self._spec_tup_gold = []
        self._spec_tup_pred = []
        self._spec_tup_level = []
        self._dev_set = dev_set
        self._epoch_num = 0

        if output_path is not None and output_path is not '':
            self._output_path = output_path+'/predictions'
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)



    def __call__(self,
                 sent=None,  sent_spec_gold=None, sent_spec_pred=None, sent_spec_level=None,
                 tup_spec_level=None, tup_spec_gold=None, tup_spec_pred=None
                 # tokens: list = None, prediction: list = None, predicate_id: list = None, confidence: list = None
                 ):
        self._spec_sent.extend(sent)
        self._spec_sent_gold.extend(sent_spec_gold)
        self._spec_sent_pred.extend(sent_spec_pred)
        self._spec_sent_level.extend(sent_spec_level)
        self._spec_tup_gold.extend(tup_spec_gold)
        self._spec_tup_pred.extend(tup_spec_pred)
        self._spec_tup_level.extend(tup_spec_level)



    def get_acc(self, y_true, y_pred):
        y_pred_new = [x for x in y_pred if x == 1]
        return len(y_pred_new)/len(y_true)

    def get_tup_recall(self, y_true, y_pred):
        n = 0
        for gold, pred in zip(y_true, y_pred):
            if gold == pred:
                n += 1
        # y_pred_new = [x for x in y_pred if x != 'N/A']
        return n/len(y_true)

    def get_metric(self, reset: bool = False):
        if reset:
            p_micro, r_micro, f1_micro, p_macro, r_macro, f1_macro, p_binary, r_binary, f1_binary, \
                easy_acc, med_acc, hard_acc = self.spec_tup_binary_score()


            self._epoch_num += 1
            self.reset()

            return {'micro_f1': f1_micro, 'micro_p': p_micro, 'micro_r': r_micro,
                    'macro_f1': f1_macro, 'macro_p': p_macro, 'macro_r': r_macro,
                    'binary_f1': f1_binary, 'binary_p': p_binary, 'binary_r': r_binary,
                    'easy_recall': easy_acc, 'med_recall': med_acc, 'hard_recall': hard_acc
                    }

        else:
            return {'micro_f1': 0.0, 'micro_p': 0.0, 'micro_r': 0.0,
                    'macro_f1': 0.0, 'macro_p': 0.0, 'macro_r': 0.0,
                    'binary_f1': 0.0, 'binary_p': 0.0, 'binary_r': 0.0,
                    'easy_recall': 0.0, 'med_recall': 0.0, 'hard_recall': 0.0
                    }

    @overrides
    def reset(self):
        self._all_sentences = []
        self._all_tokens = []
        self._all_predictions = []
        self._all_confidences = []
        self._all_predicate_id = []

        self._spec_sent = []
        self._spec_sent_gold = []
        self._spec_sent_pred = []
        self._spec_sent_level = []

        self._spec_tup_gold = []
        self._spec_tup_pred = []
        self._spec_tup_level = []


    def spec_tup_binary_score(self):
        # this section is to check sent level spec data
        # remove duplicated sentences.

        tup_true = self._spec_tup_gold
        tup_pred = self._spec_tup_pred

        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(tup_true, tup_pred, average='micro')
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(tup_true, tup_pred, average='macro')
        p_binary, r_binary, f1_binary, _ = precision_recall_fscore_support(tup_true, tup_pred, average='binary')


        easy_gold, easy_pred, med_gold, med_pred, hard_gold, hard_pred = [], [], [], [], [], []
        for gold, pred, level in zip(self._spec_tup_gold, self._spec_tup_pred, self._spec_tup_level):
            if level == 'easy':
                easy_gold.append(gold)
                easy_pred.append(pred)
            if level == 'med':
                med_gold.append(gold)
                med_pred.append(pred)
            if level == 'hard':
                hard_gold.append(gold)
                hard_pred.append(pred)


        easy_acc = self.get_acc(easy_gold, easy_pred)
        med_acc = self.get_acc(med_gold, med_pred)
        hard_acc = self.get_acc(hard_gold, hard_pred)

        return p_micro, r_micro, f1_micro, p_macro, r_macro, f1_macro, p_binary, r_binary, f1_binary,\
               easy_acc, med_acc, hard_acc



def f1(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def p(correct, pred):
    if pred == 0:
        return 0
    return correct / pred
def r(correct, gold):
    if gold == 0:
        return 0
    return correct / gold