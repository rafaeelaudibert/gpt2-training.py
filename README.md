# gpt-2 - Based on Shepherd Code

Code from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

See more details in their [blog post](https://blog.openai.com/better-language-models/).

### Some caveats

- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

## Fine tuning on custom datasets

To retrain GPT-2 117M model on a custom text dataset:

```
PYTHONPATH=src ./train.py --dataset <file|directory|glob>
```

If you want to precompute the dataset's encoding for multiple runs, you can instead use:

```
PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/encoded.npz
PYTHONPATH=src ./train.py --dataset /path/to/encoded.npz
```

### Gradient Checkpointing

https://github.com/openai/gradient-checkpointing is included to reduce the memory requirements of the model, and can be enabled by `--memory_saving_gradients`. The checkpoints are currently chosen manually (poorly) by just adding layer 10 to the 'checkpoints' collection in model.py. `--memory_saving_gradients` is enabled by default for training the 345M model.

### Validation loss

Set `--val_every` to a number of steps `N > 0`, and "validation" loss against a fixed sample of the dataset will be calculated every N steps to get a better sense of training progress. N around 200 suggested. You can set `--val_dataset` to choose a separate validation dataset, otherwise it defaults to a sample from the train dataset (so not a real cross-validation loss!).

### Optimizer

You can use SGD instead of Adam with `--optimizer sgd`. This also helps conserve memory when training the 345M model. Note: the learning rate needs to be adjusted for SGD, due to not having Adam's gradient normalization (0.0006 seems to be a good number from some experiments).
