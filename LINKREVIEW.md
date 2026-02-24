# LinkReview

- Here we collect all the works that may be useful for writing our paper
- We divide these works by topic in order to structure them

> [!NOTE]
> This review table will be updated, so it is not a final version.

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Drifting (Key Theory)** | Generative Modeling via Drifting | 2026 | Mingyang Deng et al. | [arXiv](https://arxiv.org/pdf/2602.04770v1) | [GitHub](https://github.com/tyfeld/drifting-model) | One-step generative framework that learns a deterministic linear transport map between distributions via a learned drift field. |
| **Flow-Based Generative Modeling** | Flow Matching for Generative Modeling | 2023 | Mingyang Deng et al. | [arXiv](https://arxiv.org/pdf/2210.02747) | - | Framework for training continuous normalizing flows by regressing vector fields onto conditional probability paths. |
| **Flow-Based Generative Modeling** | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | 2023 | Xingchao Liu et al. | [arXiv](https://arxiv.org/abs/2209.03003) | - | Introduces rectified flow for straight-line transport between distributions, enabling efficient generation. |
| **One-Step Generative Modeling** | Mean Flows for One-Step Generative Modeling | 2025 | Zhengyang Geng et al. | [arXiv](https://arxiv.org/pdf/2505.13447) | - | One-step generative framework modeling average velocity instead of instantaneous velocity. |
| **Video Restoration (One-Step Diffusion)** | SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training | 2025 | Jianyi Wang et al. | [arXiv](https://arxiv.org/abs/2506.05301) | - | One-step video restoration via adversarial post-training of a pre-trained diffusion transformer. |
| **One-Step Generative Modeling** | Consistency Models | 2023 | Yang Song et al. | [arXiv](https://arxiv.org/abs/2303.01469) | - | Enables single-step or few-step generation via consistency training without iterative denoising. |
| **Diffusion Acceleration** | Progressive Distillation for Fast Sampling of Diffusion Models | 2022 | Tim Salimans, Jonathan Ho | [arXiv](https://arxiv.org/abs/2202.00512) | - | Reduces multi-step diffusion sampling to few-step or single-step generation via iterative distillation. |
| **Diffusion-Based SR (Multi-Step)** | SR3: Image Super-Resolution via Iterative Refinement | 2021 | Chitwan Saharia et al. | [arXiv](https://arxiv.org/abs/2104.07636) | - | Diffusion-based SR using iterative denoising; achieves strong perceptual quality at high computational cost. |
| **Latent Diffusion for SR** | High-Resolution Image Synthesis with Latent Diffusion Models | 2022 | Robin Rombach et al. | [arXiv](https://arxiv.org/abs/2112.10752) | [GitHub](https://github.com/CompVis/latent-diffusion) | Introduces latent diffusion models, widely used for efficient high-resolution synthesis and SR pipelines. |
| **Diffusion Distillation for SR** | SinSR: Diffusion-Based Image Super-Resolution in a Single Step | 2023 | Yufei Wang et al. | [arXiv](https://arxiv.org/abs/2311.14760) | [GitHub](https://github.com/wyf0912/SinSR) | Distills multi-step diffusion SR models into a single-step network for fast inference. |
| **One-Step Diffusion for Real-World SR** | OSEDiff: One-Step Diffusion for Real-World Image Super-Resolution | 2024 | Rongyuan Wu et al. | [arXiv](https://arxiv.org/abs/2406.08177) | [GitHub](https://github.com/cswry/OSEDiff) | One-step diffusion framework for real-world SR starting directly from LQ images without initial noise. |
| **Flow-Based SR** | SRFlow: Learning the Super-Resolution Space with Normalizing Flow | 2020 | Andreas Lugmayr et al. | [arXiv](https://arxiv.org/pdf/2006.14200) | [GitHub](https://github.com/andreas128/SRFlow) | Conditional normalizing flow for SR enabling diverse and photo-realistic reconstructions via exact likelihood training. |
| **GAN-Based Super-Resolution** | SRGAN: Photo-Realistic Single Image Super-Resolution Using a GAN | 2017 | Christian Ledig et al. | [arXiv](https://arxiv.org/abs/1609.04802) | - | First GAN-based perceptual SR method introducing adversarial and perceptual losses. |
| **GAN-Based Super-Resolution** | ESRGAN: Enhanced Super-Resolution GAN | 2018 | Xintao Wang et al. | [arXiv](https://arxiv.org/abs/1809.00219) | [GitHub](https://github.com/xinntao/ESRGAN) | Improved GAN-based SR with enhanced architecture and perceptual loss. |
| **GAN-Based Super-Resolution** | Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data | 2021 | Xintao Wang et al. | [arXiv](https://arxiv.org/abs/2107.10833) | [GitHub](https://github.com/xinntao/Real-ESRGAN) | Extends ESRGAN for real-world degradation modeling and blind SR. |
