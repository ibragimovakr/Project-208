# Evaluating the Effectiveness of Generative Drifting for Single-Step Super-Resolution
<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/kisnikser/m1p-template?color=green)](https://github.com/kisnikser/m1p-template/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/kisnikser/m1p-template)](https://github.com/kisnikser/m1p-template/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/kisnikser/m1p-template.svg?color=0088ff)](https://github.com/kisnikser/m1p-template/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/kisnikser/m1p-template.svg?color=7f29d6)](https://github.com/kisnikser/m1p-template/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Kseniia Ibragimova </td>
    </tr>
    <tr>
        <td align="left"> <b> Supervisor </b> </td>
        <td> Andrei Filatov </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract
Recent advances in generative modeling have significantly improved perceptual quality in image super-resolution (SR). However, a fundamental trade-off between reconstruction fidelity and inference speed remains unresolved. Diffusion-based approaches achieve state-of-the-art visual realism but require multiple iterative denoising steps, limiting their suitability for real-time applications. In contrast, single-step models optimized with pixel-wise losses provide fast inference yet often produce overly smooth results with insufficient high-frequency detail.
In this work, we investigate the applicability of the Generative Drifting framework to the task of single-step super-resolution. We propose a conditional adaptation of the drifting mechanism that learns a direct transport map from the distribution of low-resolution images to the distribution of high-resolution images in a single forward pass. Our central hypothesis is that the learned drifting field can effectively model the residual structure between degraded and clean image manifolds, enabling high-fidelity reconstruction without iterative refinement.
We evaluate the proposed approach on standard SR benchmarks using PSNR, SSIM, and LPIPS metrics, together with inference time analysis. The method is compared against representative GAN-based architectures and distilled diffusion models to assess both perceptual quality and computational efficiency.
## Keywords
Generative Drifting, Super-Resolution, One-Step Generation, Conditional Diffusion

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
