import torch
import wandb

def interpolate(a, b):
    alpha = torch.rand(a.size(0), 1, device=a.device)
    inter = a + alpha * (b - a)
    return inter


def calculate_gradient_penalty(critic, x_real, x_fake, context):
    image = interpolate(x_real, x_fake).requires_grad_(True)
    pred = critic(image, context)
    grad = torch.autograd.grad(
        outputs=pred, inputs=image,
        grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(grad.shape[0], -1)
    norm = grad.norm(2, dim=1)
    gp = ((norm - 1.0) ** 2).mean()
    return gp


def cramer_critic(netC, real, fake2, context):
    net_real = netC(real, context)
    return torch.norm(net_real - netC(fake2, context), p=2, dim=1) - \
           torch.norm(net_real, p=2, dim=1)


def cramer_C_loss(fake1, fake2, netC, real, w, context, config):
    surrogate = torch.mean((cramer_critic(netC, real, fake2.detach(), context) -
                            cramer_critic(netC, fake1.detach(), fake2.detach(), context)) * w)

    grad_penalty = calculate_gradient_penalty(netC, real, fake1.detach(), context)

    loss = -surrogate + grad_penalty * config.losses.C.gradient_penalty

    fakes_norm = torch.norm(fake1.detach() - fake2.detach(), p=2, dim=1).detach().cpu()

    wandb.log({'critic fakes norm MIN': fakes_norm.min().item(), 'critic fakes norm MAX': fakes_norm.max().item()})
    wandb.log({'Gradient penalty': grad_penalty.item()})
    return loss


def cramer_G_loss(fake1, fake2, netC, real, w, context, config):
    surrogate = torch.mean((cramer_critic(netC, real, fake2, context) -
                            cramer_critic(netC, fake1, fake2, context)) * w)

    fakes_norm = torch.norm(fake1.detach() - fake2.detach(), p=2, dim=1).detach().cpu()
    wandb.log(
        {'generator fakes norm MIN': fakes_norm.min().item(), 'generator fakes norm MAX': fakes_norm.max().item()})

    return surrogate


def weighted_mse_loss(input, target, weight):
    return (weight.view(-1, 1) * (input - target) ** 2).mean()


def distill_G_loss(teacher, student, student_acts, teacher_acts_list, noise, w, context, config):
    with torch.no_grad():
        teacher_acts = teacher.G.get_activations(noise, context, teacher_acts_list)


    loss =  weighted_mse_loss(teacher_acts[-1], student_acts[-1], w)

    if len(student_acts) > 1:
        for i in range(len(student_acts)-1):
            loss += weighted_mse_loss(teacher_acts[i], student_acts[i],
                             torch.zeros(w.shape, device=noise.device))

    wandb.log({'G distill loss': loss.item()})

    return loss