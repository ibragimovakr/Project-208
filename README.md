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
This work investigates the applicability of the Generative Drifting framework to single-step image super-resolution. Existing super-resolution methods face a fundamental trade-off between perceptual quality and inference speed. To address this problem, we explore the Generative Drifting paradigm, which models generation as deterministic transport between probability distributions using a learned vector field. We propose a conditional formulation that learns a direct transport from low-resolution to high-resolution images in a single forward pass. Experiments show competitive reconstruction quality and improved efficiency compared with GAN-based and distilled diffusion models, measured using PSNR, SSIM, LPIPS, and inference time.
## Keywords
Generative Drifting, Super-Resolution, One-Step Generation, Conditional Diffusion


## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
