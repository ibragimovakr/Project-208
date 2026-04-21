from train_perc_baseline import load_cfg, train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.config, stochastic=True)
    cfg.output_root = "./outputs/perc_stochastic"
    cfg.zero_noise_train = False
    cfg.zero_noise_eval = False
    train(cfg)
