# Language Translation

This example shows how one might use transformers for language translation. In particular, this implementation is loosely based on the [Attention is All You Need paper](https://arxiv.org/abs/1706.03762).

## Requirements

We will need a tokenizer for our languages. Torchtext does include a tokenizer for English, but unfortunately, we will need more languages then that. We can get these tokenizers via ```spacy```

```bash
python3 -m spacy download <language>
python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm
```

Spacy supports many languages. For a full accounting of supported languages, please look [here](https://spacy.io/usage/models). This example will default from German to English.

Torchtext is also required:
```bash
pip install torchtext
```

Just running these commands will get you started:
```bash
pip install -r requirements.txt
python3 -m spacy download <language-you-want>
```

## Usage

This example contains a lot of flags that you can set to change the behavior / training of the module. You can see all of them by running:

```bash
python3 main.py --help
```

But in general, all of the settings have "sensible" defaults; however, the default translation is to translate from German to English. To *train* the model, you only need to run the following command, but there is also an example for how to use any language you want:

```bash
python main.py
python main.py src=en tgt=fr # For english to french translation
```

For model inference, you can use this command:

```bash
python main.py mode=inference mode.model_path=<path-to-model>
```

After some loading time, this will open an interactive interface where you can type in whatever sentence you are interested in translating.
