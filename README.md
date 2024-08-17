# Connecting Dreams with Visual Brainstorming Instruction (Visual Intelligence, Under-Review)
[Yasheng Sun](https://scholar.google.com/citations?user=Vrq1yOEAAAAJ&hl=en), [Bohan Li](https://arlo0o.github.io/libohan.github.io/), [Mingchen Zhuge](https://scholar.google.com/citations?user=Qnj6XlMAAAAJ&hl=en&oi=ao), [Deng-Ping Fan](https://scholar.google.com/citations?user=kakwJ5QAAAAJ&hl=en&oi=ao), [Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en&oi=ao), [Fahad Shahbaz Khan](https://scholar.google.com/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao),
[Hideki Koike](https://scholar.google.com/citations?user=Ih8cJXQAAAAJ&hl=en)
<!-- # Code and Project Coming Soon! Please Stay Tuned! -->


# Table of Content
- [News](#news)
- [Installation](#step-by-step-installation-instructions)
- [Prepare Data](#prepare-data)
- [Pretrained Model](#pretrained-model)
- [Testing](#testing)
- [License](#license)
- [Acknowledgements](#acknowledgements)


# News
- [2024/08]: Paper is on [Arxiv](http://arxiv.org/abs/2408.07317).


# Step-by-step Installation Instructions

**a. Create a conda virtual environment and activate it.**
It requires python >= 3.7 as base environment.
```shell
conda create -n sssp python=3.7 -y
conda activate sssp
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.10.0 torchvision==0.8.2 -c pytorch -c conda-forge
```

**c. Install other dependencies.**
We simply freeze our environments. Other environments might also works. Here we provide requirements.txt file for reference.
```shell
pip install -r requirements.txt
```
Note that the transformers==1.19.2 is strictly required.

# Prepare Data
- Agree to the Natural Scenes Dataset's [Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the NSD Data [Access form](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform).


# Pretrained Model
- Download [Pretrained model].


# Testing
```
bash experiments/test_language_control.sh 
```
- Please check the checkpoint path in configs/test/test_res_value_inject_idback_css15.yaml and replace it with the downloaded paths accordingly.
- Note that here we directly provides the aligned fMRI feature in fmri_vae, img_clip and text_clip directory for convenience. The overall procedure follows [fMRI-reconstruction-NSD](https://github.com/MedARC-AI/fMRI-reconstruction-NSD). You could also infer them by 
```
bash experiments/diffusion_test.sh
```

# Acknowledgements
Many thanks to these excellent open source projects: 
- [InstructPix2Pix] (https://github.com/timothybrooks/instruct-pix2pix)
- [fMRI-reconstruction-NSD] (https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
- [Versatile-Diffusion] (https://github.com/SHI-Labs/Versatile-Diffusion)
- [Stable-Diffusion] (https://github.com/CompVis/stable-diffusion)
- [InstructDiffusion] (https://github.com/cientgu/InstructDiffusion)


# Citation
If you find our paper and code useful for your research, please consider citing:
```bibtex
@misc{sun2024connectingdreamsvisualbrainstorming,
      title={Connecting Dreams with Visual Brainstorming Instruction}, 
      author={Yasheng Sun and Bohan Li and Mingchen Zhuge and Deng-Ping Fan and Salman Khan and Fahad Shahbaz Khan and Hideki Koike},
      year={2024},
      eprint={2408.07317},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2408.07317}, 
}
```