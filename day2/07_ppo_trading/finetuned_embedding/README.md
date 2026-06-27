---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4500
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
widget:
- source_sentence: 삼성생명 2020년 12월 15일 주가는?
  sentences:
  - '2021년 12월 13일 에스엘(5850, KOSPI) 주가: 종가 31,350원 (전일대비 +650원, +2.12%), 시가 30,900원,
    고가 32,250원, 저가 30,750원, 거래량 223,825주, 시가총액 15,111억원'
  - '2020년 12월 15일 삼성생명(32830, KOSPI) 주가: 종가 74,200원 (전일대비 -1,100원, -1.46%), 시가 75,300원,
    고가 75,300원, 저가 73,600원, 거래량 588,261주, 시가총액 148,400억원'
  - '2023년 7월 13일 LG디스플레이(34220, KOSPI) 주가: 종가 15,080원 (전일대비 +300원, +2.03%), 시가 14,890원,
    고가 15,260원, 저가 14,850원, 거래량 1,523,819주, 시가총액 53,958억원'
- source_sentence: 한온시스템 거래량과 시가총액
  sentences:
  - '2025년 1월 9일 코오롱인더(120110, KOSPI) 주가: 종가 27,750원 (전일대비 -300원, -1.07%), 시가 27,900원,
    고가 28,200원, 저가 27,550원, 거래량 83,212주, 시가총액 7,636억원'
  - '2020년 10월 7일 한솔케미칼(014680, KOSPI) 주가: 종가 153,000원 (전일대비 +2,000원, +1.32%), 시가
    150,000원, 고가 153,000원, 저가 149,000원, 거래량 60,650주, 시가총액 17,281억원'
  - '2022년 10월 14일 한온시스템(18880, KOSPI) 주가: 종가 7,370원 (전일대비 +160원, +2.22%), 시가 7,280원,
    고가 7,460원, 저가 7,260원, 거래량 849,296주, 시가총액 39,341억원'
- source_sentence: LG디스플레이 거래량과 시가총액
  sentences:
  - '2021년 12월 27일 한샘(009240, KOSPI) 주가: 종가 98,000원 (전일대비 +300원, +0.31%), 시가 98,100원,
    고가 98,300원, 저가 96,900원, 거래량 87,396주, 시가총액 23,063억원'
  - '2021년 2월 9일 LG디스플레이(34220, KOSPI) 주가: 종가 23,450원 (전일대비 +450원, +1.96%), 시가 23,100원,
    고가 23,950원, 저가 23,000원, 거래량 6,021,038주, 시가총액 83,907억원'
  - '2021년 5월 14일 동서(26960, KOSPI) 주가: 종가 30,400원 (전일대비 -550원, -1.78%), 시가 30,950원,
    고가 31,050원, 저가 30,200원, 거래량 163,287주, 시가총액 30,308억원'
- source_sentence: 녹십자 주식 2026년 4월 시세
  sentences:
  - '2022년 6월 27일 한국전력(015760, KOSPI) 주가: 종가 22,850원 (전일대비 +350원, +1.56%), 시가 23,000원,
    고가 23,200원, 저가 22,700원, 거래량 3,124,974주, 시가총액 146,688억원'
  - '2020년 7월 23일 아모레퍼시픽(90430, KOSPI) 주가: 종가 165,500원 (전일대비 -2,000원, -1.19%), 시가
    167,000원, 고가 167,500원, 저가 163,000원, 거래량 181,610주, 시가총액 96,748억원'
  - '2026년 4월 15일 녹십자(006280, KOSPI) 주가: 종가 148,000원 (전일대비 +4,000원, +2.78%), 시가 145,600원,
    고가 148,100원, 저가 145,300원, 거래량 30,900주, 시가총액 17,296억원'
- source_sentence: LG이노텍 주식 2023년 4월 시세
  sentences:
  - '2026년 3월 18일 한화(880, KOSPI) 주가: 종가 125,600원 (전일대비 +3,300원, +2.70%), 시가 123,600원,
    고가 126,100원, 저가 122,600원, 거래량 199,418주, 시가총액 94,148억원'
  - '2023년 4월 18일 LG이노텍(011070, KOSPI) 주가: 종가 257,000원 (전일대비 -2,000원, -0.77%), 시가
    258,000원, 고가 258,500원, 저가 253,500원, 거래량 176,153주, 시가총액 60,824억원'
  - '2024년 5월 10일 삼성에스디에스(18260, KOSPI) 주가: 종가 156,700원 (전일대비 +1,400원, +0.90%), 시가
    157,300원, 고가 157,800원, 저가 155,400원, 거래량 71,445주, 시가총액 121,251억원'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) <!-- at revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'LG이노텍 주식 2023년 4월 시세',
    '2023년 4월 18일 LG이노텍(011070, KOSPI) 주가: 종가 257,000원 (전일대비 -2,000원, -0.77%), 시가 258,000원, 고가 258,500원, 저가 253,500원, 거래량 176,153주, 시가총액 60,824억원',
    '2024년 5월 10일 삼성에스디에스(18260, KOSPI) 주가: 종가 156,700원 (전일대비 +1,400원, +0.90%), 시가 157,300원, 고가 157,800원, 저가 155,400원, 거래량 71,445주, 시가총액 121,251억원',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8649, 0.1429],
#         [0.8649, 1.0000, 0.1258],
#         [0.1429, 0.1258, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,500 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                             | positive                                                                           |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             |
  | modality | text                                                                               | text                                                                               |
  | details  | <ul><li>min: 10 tokens</li><li>mean: 14.09 tokens</li><li>max: 21 tokens</li></ul> | <ul><li>min: 65 tokens</li><li>mean: 76.96 tokens</li><li>max: 87 tokens</li></ul> |
* Samples:
  | anchor                              | positive                                                                                                                                           |
  |:------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>현대위아 2023년 6월 14일 주가는?</code> | <code>2023년 6월 14일 현대위아(011210, KOSPI) 주가: 종가 59,400원 (전일대비 -200원, -0.34%), 시가 60,000원, 고가 60,500원, 저가 59,100원, 거래량 116,606주, 시가총액 16,153억원</code> |
  | <code>현대위아(011210) 종가 정보</code>     | <code>2023년 6월 14일 현대위아(011210, KOSPI) 주가: 종가 59,400원 (전일대비 -200원, -0.34%), 시가 60,000원, 고가 60,500원, 저가 59,100원, 거래량 116,606주, 시가총액 16,153억원</code> |
  | <code>KOSPI 종목 현대위아 6월 주가</code>    | <code>2023년 6월 14일 현대위아(011210, KOSPI) 주가: 종가 59,400원 (전일대비 -200원, -0.34%), 시가 60,000원, 고가 60,500원, 저가 59,100원, 거래량 116,606주, 시가총액 16,153억원</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `warmup_steps`: 0.1
- `fp16`: True
- `gradient_checkpointing`: True
- `dataloader_drop_last`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0890 | 50   | 0.9150        |
| 0.1779 | 100  | 0.1182        |
| 0.2669 | 150  | 0.0626        |
| 0.3559 | 200  | 0.0449        |
| 0.4448 | 250  | 0.0462        |
| 0.5338 | 300  | 0.0439        |
| 0.6228 | 350  | 0.0394        |
| 0.7117 | 400  | 0.0264        |
| 0.8007 | 450  | 0.0438        |
| 0.8897 | 500  | 0.0201        |
| 0.9786 | 550  | 0.0203        |
| 1.0676 | 600  | 0.0253        |
| 1.1566 | 650  | 0.0264        |
| 1.2456 | 700  | 0.0291        |
| 1.3345 | 750  | 0.0298        |
| 1.4235 | 800  | 0.0238        |
| 1.5125 | 850  | 0.0288        |
| 1.6014 | 900  | 0.0069        |
| 1.6904 | 950  | 0.0217        |
| 1.7794 | 1000 | 0.0193        |
| 1.8683 | 1050 | 0.0257        |
| 1.9573 | 1100 | 0.0162        |
| 2.0463 | 1150 | 0.0159        |
| 2.1352 | 1200 | 0.0389        |
| 2.2242 | 1250 | 0.0052        |
| 2.3132 | 1300 | 0.0342        |
| 2.4021 | 1350 | 0.0177        |
| 2.4911 | 1400 | 0.0272        |
| 2.5801 | 1450 | 0.0198        |
| 2.6690 | 1500 | 0.0119        |
| 2.7580 | 1550 | 0.0278        |
| 2.8470 | 1600 | 0.0160        |
| 2.9359 | 1650 | 0.0188        |


### Training Time
- **Training**: 2.9 minutes

### Framework Versions
- Python: 3.14.4
- Sentence Transformers: 5.6.0
- Transformers: 5.12.1
- PyTorch: 2.12.1+cu130
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->