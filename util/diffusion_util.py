import numpy as np
import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    alpha = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
    return alpha


def get_para_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timestep, device=None):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timestep,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timestep, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timestep, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timestep, 1, num_diffusion_timestep, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timestep)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timestep,)

    betas = torch.tensor(betas, device=device).float()
    alphas = (1. - betas)
    alphas_bar = (1. - betas).cumprod(dim=0)

    return betas, alphas, alphas_bar


# Qt = alphas_bar * I + (1 - alphas_bar) * K
def get_Qt_weight(alphas_bar, noise, batch, device, n_class=20):
    # [N,20,20]
    Qt_weight = [bar_t * torch.eye(n_class, device=device) + (1 - bar_t) * noise for bar_t in
                 alphas_bar]

    Qt_weight = torch.stack(Qt_weight).float()
    Qt_weight = Qt_weight.index_select(0, batch)

    return Qt_weight


def get_Qt_weight_batch(alphas_bar, batch_noise, batch, device, n_class=20):
    Qt_weight_list = []
    for index in range(len(batch_noise)):
        bar_t = alphas_bar[index]
        noise_t = batch_noise[index]

        Qt_weight = bar_t * torch.eye(n_class, device=device) + (1 - bar_t) * noise_t
        Qt_weight_list.append(Qt_weight)

    Qt_weight_list = torch.stack(Qt_weight_list).float()
    Qt_weight_list = Qt_weight_list.index_select(0, batch)

    return Qt_weight_list


# KL(P||Q) = E_x[P(x)log(P(x)/Q(x))]
def multinomial_kl(prob1, prob2):
    # kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    prob1 = prob1.softmax(dim=-1)
    prob2 = prob2.softmax(dim=-1)
    kl = (prob1 * torch.log(prob1 / prob2)).sum(dim=-1)

    return kl.mean()


def token_aa_acc(pred, real, device):
    y_pred = torch.argmax(pred, dim=-1)
    y_real = torch.argmax(real, dim=-1)
    score = torch.sum(torch.tensor(y_pred == y_real, device=device)) / len(y_pred)
    return score
