import argparse


def setting_goGoTriple_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024,
                        help='seed for model initialization')

    parser.add_argument('--out_log_dir', default='data/output/log',
                        help='Path to log file.')

    parser.add_argument('--out_checkpoint_dir', default='data/output/go_go_triple',
                        help='Path to checkpoint file.')

    parser.add_argument('--save_top_k', default=10,
                        help='save_top_k for train.')

    parser.add_argument('--save_mode', default="min",
                        help='save_mode for train.')

    parser.add_argument('--gpus', default=[0],
                        help='gpus for train.')

    parser.add_argument('--n_max_epochs', default=3000,
                        help='max_epochs for train.')

    parser.add_argument('--batch_size', default=2048,
                        help='batch_sizes for train.')

    parser.add_argument('--max_score', default=12,
                        help='max_score for transE.')

    parser.add_argument('--learning_rate', default=1e-4,
                        help='learning_rate for training.')

    return parser.parse_args()


def setting_pepGoTriple_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024,
                        help='seed for model initialization')

    parser.add_argument('--out_log_dir', default='data/output/log',
                        help='Path to log file.')

    parser.add_argument('--out_checkpoint_dir', default='data/output/pep_go_triple',
                        help='Path to checkpoint file.')

    parser.add_argument('--save_top_k', default=10,
                        help='save_top_k for train.')

    parser.add_argument('--save_mode', default="min",
                        help='save_mode for train.')

    parser.add_argument('--gpus', default=[0],
                        help='gpus for train.')

    parser.add_argument('--n_max_epochs', default=300,
                        help='max_epochs for train.')

    parser.add_argument('--goGoTriple_checkpoint',
                        help='loading the goGoTriple_checkpoint_checkpoint for pre-training; '
                             'like data/output/go_go_triple/go_go_triple-epoch=XX-avg_loss=XX.ckpt.')

    parser.add_argument('--learning_rate_seq', default=1e-4,
                        help='learning_rate for peptide sequence.')

    parser.add_argument('--learning_rate_go', default=5e-5,
                        help='learning_rate for go term.')

    parser.add_argument('--batch_size', default=1024,
                        help='batch_sizes for train.')

    parser.add_argument('--max_score', default=10,
                        help='max_score for transE.')

    parser.add_argument('--fasta_type', default="uniport",
                        help='sequence type for fasta file, amp or uniport.')

    return parser.parse_args()


def setting_pepPathogenTriple_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024,
                        help='seed for model initialization')

    parser.add_argument('--out_log_dir', default='data/output/log',
                        help='Path to log file.')

    parser.add_argument('--out_checkpoint_dir', default='data/output/pep_pathogen_triple',
                        help='Path to checkpoint file.')

    parser.add_argument('--save_top_k', default=10,
                        help='save_top_k for train.')

    parser.add_argument('--save_mode', default="min",
                        help='save_mode for train.')

    parser.add_argument('--gpus', default=[0],
                        help='gpus for train.')

    parser.add_argument('--n_max_epochs', default=2000,
                        help='max_epochs for train.')

    parser.add_argument('--pepGoTriple_checkpoint',
                        help='loading the pepGoTriple_checkpoint for pre-training; '
                             'like data/output/pep_go_triple/pep_go_triple-epoch=XX-avg_loss=XX-pep_avg_loss=XX-go_avg_loss=XX.ckpt.')

    parser.add_argument('--learning_rate', default=1e-4,
                        help='learning_rate for model.')

    parser.add_argument('--learning_rate_pre', default=5e-5,
                        help='learning_rate_pre for the pretrained component.')

    parser.add_argument('--batch_size', default=1024,
                        help='batch_sizes for train.')

    parser.add_argument('--max_score', default=10,
                        help='max_score for transE.')

    parser.add_argument('--fasta_type', default="amp",
                        help='sequence type for fasta file, amp or uniport.')
    return parser.parse_args()


def setting_pepGen_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024,
                        help='seed for model initialization')

    parser.add_argument('--out_log_dir', default='data/output/log',
                        help='Path to log file.')

    parser.add_argument('--out_checkpoint_dir', default='data/output/pepSeqCondDiff',
                        help='Path to checkpoint file.')

    parser.add_argument('--save_top_k', default=10,
                        help='save_top_k for train.')

    parser.add_argument('--save_mode', default="max",
                        help='save_mode for train.')

    parser.add_argument('--gpus', default=[0],
                        help='gpus for train.')

    parser.add_argument('--n_max_epochs', default=5000,
                        help='max_epochs for train.')

    parser.add_argument('--pepPathogenTriple_checkpoint',
                        help='loading the pepPathogenTriple_checkpoint for pre-training;'
                             'data/output/pep_pathogen_triple/pep_pathogen_Triple-epoch=XX-avg_loss=XX.ckpt.')

    parser.add_argument('--learning_rate', default=1e-4,
                        help='learning_rate for model.')

    parser.add_argument('--learning_rate_pre', default=5e-5,
                        help='learning_rate_pre for the pretrained component.')

    parser.add_argument('--condition_weight', default=2,
                        help='weight for the unconditional/conditional components.')

    parser.add_argument('--batch_size', default=512,
                        help='batch_sizes for train.')

    parser.add_argument('--n_timestep', default=1000,
                        help='n_timestep for diffusion process.')

    parser.add_argument('--beta_schedule', default="linear",
                        help='beta_schedule for diffusion parameter.')

    parser.add_argument('--beta_start', default=1.e-4,
                        help='beta_start for diffusion parameter.')

    parser.add_argument('--beta_end', default=2e-2,
                        help='beta_end for diffusion parameter.')

    return parser.parse_args()
