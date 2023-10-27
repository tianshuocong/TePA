# 🎃TePA

This repository contains the PyTorch implementation of the paper "[Test-Time Poisoning Attacks Against Test-Time Adaptation Models](https://arxiv.org/abs/2308.08505)" by [Tianshuo Cong](https://tianshuocong.github.io/), [Xinlei He](https://xinleihe.github.io/), [Yun Shen](https://scholar.google.com/citations?hl=en&user=Gx_JJ6cAAAAJ), and [Yang Zhang](https://yangzhangalmo.github.io/).
In particular, we propose the first *test-time* poisoning attack against four mainstream test-time adaptation methods, including TTT, DUA, TENT, and RPL. Here is the workflow of TePA:

<div align="center">
<img width="80%" alt="The workflow of TePA" src="fig/workflow.png">
</div>

## Citation ☺️
Thanks for your interest in our paper, please feel free to give us a ⭐️ and cite us through:
```bibtex
@inproceedings{cong2024tepa,
  title={Test-Time Poisoning Attacks Against Test-Time Adaptation Models},
  author={Tianshuo Cong and Xinlei He and Yun Shen and Yang Zhang},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2024}
}
```

## Requirements 🔧
TePA depends on the following requirements:
- Basic: PyTorch 1.11.0, Python 3.8, Cuda 11.3
- Others:
  - TTT: https://github.com/yueatsprograms/ttt_cifar_release
  - TENT: https://github.com/DequanWang/tent
  - DUA: https://github.com/jmiemirza/DUA

## Baseline 🎯
- First, we should check the utility of the frozen target model, and the utility of the TTT (using clean i.i.d. samples), run the following code, and then we can get the results of Figure 4 and Table 1 of our paper.
```
python TTT/utility.py
```

## Poison TTA-models 🦠
```
python TTT/poison_ttt.py
```
Then, we can get the results of Figure 5-8.

<div align="center">
<img width="80%" alt="The workflow of TePA" src="fig/poison_result.png">
</div>
