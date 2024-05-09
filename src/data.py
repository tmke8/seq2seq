from collections.abc import Callable, Generator, Iterable, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, multi30k
from torchtext.vocab import Vocab, build_vocab_from_iterator

from src.config import Language, Options


# Turns an iterable into a generator
def _yield_tokens(
    iterable_data: Dataset[tuple[str, str]], tokenizer: Callable[[str], list[str]], src: bool
) -> Generator[list[str], None, None]:
    # Iterable data stores the samples as (src, tgt)
    # so this will help us select just one language or the other
    index = 0 if src else 1

    for data in iterable_data:
        yield tokenizer(data[index])


# Get data, tokenizer, text transform, vocab objs, etc. Everything we
# need to start training the model
def get_data(
    opts: Options,
) -> tuple[
    DataLoader[tuple[str, str]],
    DataLoader[tuple[str, str]],
    Vocab,
    Vocab,
    Callable[[str], Tensor],
    Callable[[str], Tensor],
    dict[str, int],
]:
    src_lang = opts.src
    tgt_lang = opts.tgt

    multi30k.URL["train"] = (
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    )
    multi30k.URL["valid"] = (
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    )

    # Define a token "unkown", "padding", "beginning of sentence", and "end of sentence"
    special_symbols = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}

    # Get training examples from torchtext (the multi30k dataset)
    train_iterator: Dataset[tuple[str, str]] = Multi30k(
        split="train", language_pair=(src_lang.code, tgt_lang.code)
    )
    valid_iterator: Dataset[tuple[str, str]] = Multi30k(
        split="valid", language_pair=(src_lang.code, tgt_lang.code)
    )

    # Grab a tokenizer for these languages
    src_tokenizer: Callable[[str], list[str]] = get_tokenizer("spacy", src_lang.spacy_model)
    tgt_tokenizer: Callable[[str], list[str]] = get_tokenizer("spacy", tgt_lang.spacy_model)

    # Build a vocabulary object for these languages
    src_vocab = build_vocab_from_iterator(
        _yield_tokens(train_iterator, src_tokenizer, src=True),
        min_freq=1,
        specials=list(special_symbols.keys()),
        special_first=True,
    )

    tgt_vocab = build_vocab_from_iterator(
        _yield_tokens(train_iterator, tgt_tokenizer, src=False),
        min_freq=1,
        specials=list(special_symbols.keys()),
        special_first=True,
    )

    src_vocab.set_default_index(special_symbols["<unk>"])
    tgt_vocab.set_default_index(special_symbols["<unk>"])

    # Function to add BOS/EOS and create tensor for input sequence indices
    def _tensor_transform(token_ids: Sequence[int]) -> Tensor:
        return torch.cat(
            (
                torch.tensor([special_symbols["<bos>"]]),
                torch.tensor(token_ids),
                torch.tensor([special_symbols["<eos>"]]),
            )
        )

    def src_lang_transform(inputs: str) -> Tensor:
        return _tensor_transform(src_vocab(src_tokenizer(inputs)))

    def tgt_lang_transform(inputs: str) -> Tensor:
        return _tensor_transform(tgt_vocab(tgt_tokenizer(inputs)))

    # Now we want to convert the torchtext data pipeline to a dataloader. We
    # will need to collate batches
    def _collate_fn(batch: Iterable[tuple[str, str]]) -> tuple[Tensor, Tensor]:
        src_batch: list[Tensor] = []
        tgt_batch: list[Tensor] = []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_lang_transform(src_sample.rstrip("\n")))
            tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))

        src_batch_ = pad_sequence(src_batch, padding_value=special_symbols["<pad>"])
        tgt_batch_ = pad_sequence(tgt_batch, padding_value=special_symbols["<pad>"])
        return src_batch_, tgt_batch_

    # Create the dataloader
    train_dataloader = DataLoader(train_iterator, batch_size=opts.batch, collate_fn=_collate_fn)
    valid_dataloader = DataLoader(valid_iterator, batch_size=opts.batch, collate_fn=_collate_fn)

    return (
        train_dataloader,
        valid_dataloader,
        src_vocab,
        tgt_vocab,
        src_lang_transform,
        tgt_lang_transform,
        special_symbols,
    )


# A small test to make sure our data loasd in correctly
if __name__ == "__main__":
    opts = Options(src=Language.en, tgt=Language.de, batch=128)

    (
        train_dl,
        valid_dl,
        src_vocab,
        tgt_vocab,
        src_lang_transform,
        tgt_lang_transform,
        special_symbols,
    ) = get_data(opts)

    print(f"{opts.src} vocab size: {len(src_vocab)}")
    print(f"{opts.src} vocab size: {len(tgt_vocab)}")
