from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch
import torch.utils.checkpoint
from torch import nn
import warnings
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    NextSentencePredictorOutput
)

class RobertaForNextSentencePrediction(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.cls = RobertaOnlyNSPHead(config)
        self.roberta.config.type_vocab_size = 2
        single_emb = self.roberta.embeddings.token_type_embeddings
        self.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        self.roberta.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring). Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score