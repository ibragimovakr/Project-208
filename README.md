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
- [Paper](paper/ibragimova_paper.pdf)
- [Slides]()

## Abstract
This paper investigates Generative Drifting for one-step single-image super-resolution in the one-to-many setting. Super-resolution is inherently ill-posed. A single low-resolution image corresponds to multiple plausible high-resolution reconstructions. Nevertheless, efficient deterministic methods often produce averaged and over-smoothed results. To address this limitation, we extend Generative Drifting to conditional super-resolution. We propose a stochastic one-step model that maps a low-resolution input and random noise directly to plausible high-resolution outputs. The method performs drifting in feature space, encouraging samples to move toward admissible high-resolution targets while avoiding collapse to a single reconstruction. Experiments show that conditional drifting improves the fidelity--diversity trade-off over a matched stochastic pixel/LR-consistency baseline. We also find that this trade-off strongly depends on the geometry of the feature space used to compute the drifting field. These results suggest that conditional drifting is an effective mechanism for learning LR-consistent one-to-many reconstructions in a single forward pass.

## Keywords
Single-image super-resolution, conditional generation, one-step generation, generative drifting, stochastic restoration


## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
