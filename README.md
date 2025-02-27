# ShortcutFM
> Official Codebase for [*ShortcutFM*](to be added).

## Dataset
Prepare datasets and put them under the `datasets` folder. Take `datasets/CommonsenseConversation/train.jsonl` as an example. We use four datasets in our paper.

| Task | Datasets | Source |
|-|-|-|
| Open-domain Dialogue | CommonsenseConversation | [download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing)|
| Question Generation | Quasar-T |[download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing) |
| Text Simplification | Wiki-alignment | [download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing)|
| Paraphrase | QQP-Official |[download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing) |
| Machine Translation | iwslt14-de-en | [download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing)|

## Training
For Non-MT (Machine Translation) tasks, run:
```bash
# qqp:
uv run python -m shortcutfm configs/qqp.yaml
# others: modify learning_steps, dataset, data_dir, notes
```

For MT tasks, run:
```bash
uv run python -m shortcutfm configs/mt.yaml
```

The trained checkpoints are provided here: [link of ckpt](to be added)

## Decoding
```bash
uv run python -m shortcutfm configs/qqp-decode.yaml
# core parameters: step and td
```

## Evaluation
```bash
uv run python -m shortcutfm configs/qqp-eval.yaml
# you can eval single file or multiple file which are in the same folder (mbr in default)
```

## Citation
Please add the citation if our paper or code helps you.
```tex

```





