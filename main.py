import logging  # Logging tools
from collections.abc import Callable
from pathlib import Path  # Creating and finding files/directories
from time import monotonic  # Track how long an epoch takes

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Transformer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm  # For fancy progress bars

from src.config import InferenceConfig, Options, TrainingConfig  # Configuration options
from src.data import get_data  # Loading data and data preprocessing
from src.model import Translator, create_mask  # Our model


def run(opts: Options, cfg: TrainingConfig) -> None:
    # Set up logging
    opts.logging_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir / "log.txt", level=logging.INFO)

    # This prints it to the screen as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(f"Translation task: {opts.src} -> {opts.tgt}")
    logging.info(f"Using device: {opts.device}")

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    train_dl, valid_dl, src_vocab, tgt_vocab, _, _, special_symbols = get_data(opts)

    logging.info("Loaded data")

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    logging.info(f"{opts.src} vocab size: {src_vocab_size}")
    logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")

    # Create model
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout,
    ).to(opts.device)

    logging.info("Model created... starting training!")

    # Set up our learning tools
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = 1e6

    for idx, epoch in enumerate(range(1, cfg.epochs + 1)):
        start_time = monotonic()
        train_loss = train(model, train_dl, loss_fn, optim, special_symbols, opts)
        epoch_time = monotonic() - start_time
        val_loss = validate(model, valid_dl, loss_fn, special_symbols, opts)

        # Once training is done, we want to save out the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir / "best.pt")

        torch.save(model.state_dict(), opts.logging_dir / "last.pt")

        logger.info(
            f"Epoch: {epoch}\n\tTrain loss: {train_loss:.3f}\n\tVal loss: {val_loss:.3f}\n"
            f"\tEpoch time = {epoch_time:.1f} seconds\n"
            f"\tETA = {epoch_time*(cfg.epochs-idx-1):.1f} seconds"
        )


def train(
    model: Translator,
    train_dl: DataLoader[tuple[str, str]],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optim: Optimizer,
    special_symbols: dict[str, int],
    opts: Options,
) -> float:
    """Train the model for 1 epoch."""
    # Object for accumulating losses
    losses = 0.0

    # Put model into inference mode
    model.train()
    for src, tgt in tqdm(train_dl, ascii=True):
        src = src.to(opts.device)
        tgt = tgt.to(opts.device)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, special_symbols["<pad>"], opts.device
        )

        # Pass into model, get probability over the vocab out
        logits = model(
            src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask
        )

        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()

        # Get original shape back
        tgt_out = tgt[1:, :]

        # Compute loss and gradient over that loss
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        # Step weights
        optim.step()

        # Accumulate a running loss for reporting
        losses += loss.item()

        if opts.dry_run:
            break

    # Return the average loss
    return losses / len(list(train_dl))


# Check the model accuracy on the validation dataset
def validate(
    model: Translator,
    valid_dl: DataLoader[tuple[str, str]],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    special_symbols: dict[str, int],
    opts: Options,
) -> float:
    # Object for accumulating losses
    losses = 0.0

    # Turn off gradients a moment
    model.eval()

    for src, tgt in tqdm(valid_dl):
        src = src.to(opts.device)
        tgt = tgt.to(opts.device)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, special_symbols["<pad>"], opts.device
        )

        # Pass into model, get probability over the vocab out
        logits = model(
            src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask
        )

        # Get original shape back, compute loss, accumulate that loss
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    # Return the average loss
    return losses / len(list(valid_dl))


# Opens an user interface where users can translate an arbitrary sentence
def inference(opts: Options, model_path: Path) -> None:
    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    _, _, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Create model
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout,
    ).to(opts.device)

    # Load in weights
    model.load_state_dict(torch.load(model_path))

    # Set to inference
    model.eval()

    # Accept input and keep translating until they quit
    while True:
        print("> ", end="")

        sentence = input()

        # Convert to tokens
        src = src_transform(sentence).view(-1, 1)
        num_tokens = src.shape[0]

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        # Decode
        tgt_tokens = greedy_decode(
            model,
            src,
            src_mask,
            max_len=num_tokens + 5,
            start_symbol=special_symbols["<bos>"],
            end_symbol=special_symbols["<eos>"],
            opts=opts,
        ).flatten()

        # Convert to list of tokens
        output_as_list = list(tgt_tokens.cpu().numpy())

        # Convert tokens to words
        output_list_words = tgt_vocab.lookup_tokens(output_as_list)

        # Remove special tokens and convert to string
        translation = " ".join(output_list_words).replace("<bos>", "").replace("<eos>", "")

        print(translation)


# Function to generate output sequence using greedy algorithm
def greedy_decode(
    model: Translator,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    opts: Options,
) -> Tensor:
    # Move to device
    src = src.to(opts.device)
    src_mask = src_mask.to(opts.device)

    # Encode input
    memory = model.encode(src, src_mask)

    # Output will be stored here
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(opts.device)

    # For each element in our translation (which could range up to the maximum translation length)
    for _ in range(max_len - 1):
        # Decode the encoded representation of the input
        memory = memory.to(opts.device)
        tgt_mask = (
            Transformer.generate_square_subsequent_mask(ys.size(0), opts.device).type(torch.bool)
        ).to(opts.device)
        out = model.decode(ys, memory, tgt_mask)

        # Reshape
        out = out.transpose(0, 1)

        # Covert to probabilities and take the max of these probabilities
        prob = model.ff(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Now we have an output which is the vector representation of the translation
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break

    return ys


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(hydra_config: DictConfig) -> None:
    opts = instantiate(hydra_config, _convert_="object")
    assert isinstance(opts, Options)
    match opts.mode:
        case TrainingConfig():
            run(opts, opts.mode)
        case InferenceConfig(model_path):
            inference(opts, model_path)
        case _:
            raise ValueError(f"Unsupported mode: {opts.mode}")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    # The name of the main config has to appear in `conf/config.yaml`.
    cs.store(node=Options, name="config_schema")
    # variants
    cs.store(node=InferenceConfig, name="inference", group="mode")
    cs.store(node=TrainingConfig, name="train", group="mode")
    main()
