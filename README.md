# Conditional Generative Drifting for One-Step One-to-Many Super-Resolution
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
This work investigates the applicability of the Generative Drifting framework to one-step image super-resolution in the one-to-many setting. A single low-resolution image may correspond to multiple plausible high-resolution reconstructions, yet most efficient super-resolution methods are trained deterministically and therefore tend to produce averaged, over-smoothed outputs. To address this limitation, we explore the Generative Drifting paradigm, which models generation through a learned drift field and enables single-step inference. We propose a conditional stochastic formulation that learns a direct mapping from a low-resolution image and noise to plausible high-resolution images in a single forward pass. The proposed method defines drifting in residual feature space, encouraging generated samples to move toward plausible high-resolution targets while preventing collapse to a single solution. Experiments evaluate reconstruction quality, perceptual realism, and diversity using PSNR, SSIM, LPIPS, and one-to-many consistency metrics.
## Keywords
Generative Drifting, Super-Resolution, One-Step Generation


## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
