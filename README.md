# From Speculation Detection to Trustworthy Relational Tuples in Information Extraction

## Introduction
About SpecTup: we propose to study speculations in OIE tuples and determine whether a tuple is speculative. We formally define the research problem of tuple-level speculation detection. We then conduct detailed analysis on the LSOIE dataset which provides labels for speculative tuples. Lastly, we propose a baseline model SpecTup for this new research task.

## SpecTup Model
### Installation Instructions

Use a python-3.7 environment and install the dependencies using,
```
pip install -r requirements.txt
```

### Preparing traing dataset
The train and test dataset is compressed in the path
```data/ ```.
Please unzip it in the same path before training.

### Running the code

```
python allennlp_run.py --config config/spec_tup.json --epoch 1 --batch 32 --model trained_model/spec_tup
```

Arguments:
- config: configuration file containing all the parameters for the model
- model:  path of the directory where the model will be saved
- epoch:  number of epoch for training
- batch:  number of instances per batch



## Citing
If you use this code in your research, please cite:

```
@inproceedings{dong-etal-2023-speculation,
    title = "From Speculation Detection to Trustworthy Relational Tuples in Information Extraction",
    author = "Dong, Kuicai  and
      Sun, Aixin  and
      Kim, Jung-jae  and
      Li, Xiaoli",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.886",
    doi = "10.18653/v1/2023.findings-emnlp.886",
    pages = "13287--13299",
    abstract = "Speculation detection is an important NLP task to identify text factuality. However, the extracted speculative information (e.g., speculative polarity, cue, and scope) lacks structure and poses challenges for direct utilization in downstream tasks. Open Information Extraction (OIE), on the other hand, extracts structured tuples as facts, without examining the certainty of these tuples. Bridging this gap between speculation detection and information extraction becomes imperative to generate structured speculative information and trustworthy relational tuples. Existing studies on speculation detection are defined at sentence level; but even if a sentence is determined to be speculative, not all factual tuples extracted from it are speculative. In this paper, we propose to study speculations in OIE tuples and determine whether a tuple is speculative. We formally define the research problem of tuple-level speculation detection. We then conduct detailed analysis on the LSOIE dataset which provides labels for speculative tuples. Lastly, we propose a baseline model SpecTup for this new research task.",
}

```

## Contact
In case of any issues, please send a mail to
```kuicai001 (at) e (dot) ntu (dot) edu (dot) sg```