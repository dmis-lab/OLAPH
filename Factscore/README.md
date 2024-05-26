# FActScore

This section explains how to measure FactScore on the K-QA Golden Dataset. There are two methods to score the model: using Wikipedia or biomedical texts (PubMed, PMC, etc.).
For Wikipedia, we have mapped the topic entities for each question that corresponds to the Wikipedia title. These titles are used as queries in the Wikipedia dump to retrieve documents.
For biomedical texts, we used MedCPT to retrieve the top 100 evidences, utilizing the question as a query. Note that we used the top 20 evidences to score the model.

## Install

Make a new Python 3.7+ environment using `virtualenv` or `conda`.

```bash
pip install --upgrade factscore
python -m spacy download en_core_web_sm
```

## Download the data

```bash
python -m factscore.download_data
```

This command doe downloads the knowledg source (wikipedia dump) and example data.

**Optional flags**:
- `--data_dir`: directory to store the knowledge source and example data. `.cache/factscore` by default.

**Troubleshooting**:
- If you get a `ERROR 429: Too Many Requests` error while downloading the DB file, please download the DB from [this Google Drive link](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view?usp=sharing) and place it under `--data_dir` (`.cache/factscore` by default).

## Preprocess a prediction file
```bash
python preprocess.py \
--input_file {input_file} \
--output_file {output_file} \
--top_k {top_k} \
```

- `--input_file`: The path to the prediction file to be processed.
- `--output_file`: The path where the preprocessed file will be saved.
- `--top_k`: The number of top results to measure using biomedical texts. Default is 20.
- `--evidence_file`: The path to the evidence file. Default is `kqa_golden_test_evidence_factscore.jsonl`.

## Running FActScore using a command line

We expect running FActScore costs about $1 of the API cost per 100 sentences. For instance, if you have 100 generations, each with 5 sentences on average, it costs $5 in total.

```bash
python -m factscore.factscorer \
--input_path {input_path} \
--model_name {estimator_name} \
--openai_key {openai_key} \
--knowledge_source {knowledge_source} \
```

- `--input_path`: The path to the preprocessed prediction file.
- `--model_name`: Default is `retrieval+ChatGPT`. (Other supported configurations include `retrieval+llama+npm` and `retrieval+ChatGPT+npm`, and `retrieval+llama`. These alternatives have not been tested extensively, so we recommend using the default configuration.)
- `--openai_key`: '.txt' file containing OpenAI API Key.
- `--knowledge_source`: `enwiki-20230401` for using (Wikipedia Dump - 2023/04/01) as a knowledge source / `biomedical` for using biomedical texts as a knowledge source.

**Optional flags**:
- `--data_dir`: Directory containing knowledge source, etc. `.cache/factscore` by default.
- `--cache_dir`: Directory containing cache from API/models. `.cache/factscore` by default.
- `--use_atomic_facts`: If specified, it uses model-generated atomic facts released as part of our data instead of running the atomic fact generator. This will allow reproducing our results with no (or little if it still uses ChatGPT) cost. You can't specify it if you are running new model generations.
- `--gamma`: A hyperparameter for length penalty. `10` by default. It penalizes the score if the number of facts is less than `gamma`. `10` roughly corresponds to 2 sentences, so would penalize if the generation has less than 2 sentences. Usually, this would not change the ranking between systems unless some systems generate overly short responses all the time (e.g., models trained on NLP datasets without long-form generation tasks may do so). If you would like to turn off the length penalty completely, specify `--gamma 0`.
- `--n_samples`: If specified, it runs the model on a subset of the data.
- `--verbose`: If specified, it shows the progress bar.
- `--print_rate_limit_error`: It specified, it prints out rate limit errors from OpenAI API.
- `--cost_estimate`: This flag decides the type of OpenAI API cost estimation that we provide before calling it. It can be `"consider_cache"` (default) or `"ignore_cache"`.
- `--abstain_detection`: This flag optionally enables automatic detection of abstained responses. By default this is disabled, but it is recommended to add your own function tailored to your model. The currently supported detectors are `"generic"` and `"perplexity_ai"`, and their implementations can be found in [`factscore/abstain_detection.py`](factscore/abstain_detection.py). There are two methods to add your own abstain function: a) clone our GitHub repository to install `factscore` locally (`pip install --editable .`), and then add your function to [`factscore/abstain_detection.py`](factscore/abstain_detection.py) directly; b) process your abstain detection outside our package, and use empty strings in the `output` key for the JSONL file used in `--input_path`.
- `--knowledge_source`: In case the default knowledge source (Wikipedia - 2023/04/01) will not be used, preprocess it using the [instructions below](#To-use-a-custom-knowledge-source), and then specify the knowledge_source name under this flag.



## Reference
```
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
```