# PLM

## File List

| file | description |
| --- | --- |
| [data_collator.py](data_collator.py) | data collator for emoji masking |
| [emoji_list](emoji_list) | unicode of emojis, copied from [emoji](pypi.org/emoji) |
| [mlm_emoji.py](mlm_emoji.py) | training code for masked emoji |
| [run_mlm.sh](run_mlm.sh) | shell script for training masked emoji|
| [RobertaForNextSentencePrediction.py](RobertaForNextSentencePrediction.py) | roberta nsp model |
| [run_nsp.py](run_nsp.py) | training code for nsp |
| [tokenizer.py](tokenizer.py) | roberta tokenizer |

## Masked Emoji

architecture: 和roberta相同

具体 mask 方法：选择所有的emoji，其中的80%保持[MASK]，10%替换成其他的token，10%保持不变。

1. model: [transformers.RobertaForMaskedLM](transformers.RobertaForMaskedLM)
2. [data_collator.py](data_collator.py) 文件中描述了对 emoji 做 mask 的过程。改编自[DataCollatorForWholeWordMask](https://huggingface.co/transformers/_modules/transformers/data/data_collator.html#DataCollatorForWholeWordMask)。
    * `DataCollatorForEmojiMask`类中，输入为tokenizer类，用于返还 mask 后的句子和对应的 labels。
    * 其中`find_all_emojis`函数用于配对句子中的 emoji token。
3. train: [mlm_emoji.py](mlm_emoji.py) 和[run_mlm.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py)类似

## Next Sentence Prediction

architecture: 和 BERT 中的 NextSentencePrediction 类似。

1. [RobertaForNextSentencePrediction.py](RobertaForNextSentencePrediction.py) 中定义了基于 Roberta 的 NSP model。改编自[BertForNextSentencePrediction
](https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForNextSentencePrediction)。

2. [tokenizer.py](tokenizer.py) 中定义了 tokenizer，和原 [RobertaTokenizer](https://huggingface.co/transformers/model_doc/roberta.html#transformers.RobertaTokenizer) 不同的地方在`create_token_type_ids_from_sequences`函数。

BERT输入时的 input embeddings = token embeddings + segment embeddings + position embeddings，但由于 Roberta 没有 NSP Model，输入时的 segment embeddings 不考虑两句子间的关系，均设置为[0, 0, ..., 0]。在`transformers`库的tokenizer和model中，为了简化这一步骤，将 input embeddings 设置为两层，去掉了 input embeddings。

由于我们的 NSP model 需要考虑 segment embeddings，故在 `create_token_type_ids_from_sequences` 函数中将修改为`[前句]*0 + [后句]*1`的形式。

3. [run_nsp.py](run_nsp.py)：和 [run_mlm.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py) 类似，和原run_mlm不同的是，增加了一层 input embeddings 作为输入，增加的具体原因同2。