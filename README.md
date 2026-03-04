# Generative Drifting for One-Step Super-Resolution
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
This work studies the applicability of the Generative Drifting framework to one-step image super-resolution. Generative Drifting models generation as a deterministic transport between probability distributions, where a learned vector field shifts samples from a source distribution toward a target one. We propose a conditional drifting formulation that learns a direct transport map from low-resolution images to high-resolution images in a single forward pass. The approach aims to capture residual structure between degraded and clean image manifolds without iterative refinement. The method is evaluated on standard super-resolution benchmarks using PSNR, SSIM, and LPIPS, and compared with GAN-based and distilled diffusion models in terms of reconstruction quality and inference time.  
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
