import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from data import DataHandler, DotDict
import subprocess
import wandb
import seaborn as sns
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
from model import RICHGAN
from metrics import calculate_rocauc
from losses import cramer_C_loss, cramer_G_loss

def load_config():
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)

    return config


def log_histograms(true, gen, e=-1):
    true = true.numpy()
    gen = gen.numpy()
    fig, axs = plt.subplots(1, 5, figsize=(30, 5))
    ftrs = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
    for i in range(5):
        sns.distplot(true[:, i], ax=axs[i], kde=False, bins=100, label='True data')
        sns.distplot(gen[:, i], ax=axs[i], kde=False, bins=100, label='Generated data')
        #axs[i].set_ylim(0, 0.43)
        axs[i].title.set_text(ftrs[i])
        axs[i].legend()
    wandb.log({'epoch':e, 'hist':wandb.Image(plt)})
    plt.show()


if __name__ == '__main__':
    CONFIG_PATH = "./config.yaml"

    config = load_config()

    config = DotDict(config)
    fulldata = DataHandler(config=config)

    bashCommand = "wandb login <key>"

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    wandb.init(project='...', entity='...')

    model = RICHGAN(config).to(config.utils.device)

    g_update_freq = 15

    lfunc = lambda epoch: 0.95 ** epoch
    g_scheduler = optim.lr_scheduler.LambdaLR(model.optim_g, lr_lambda=lfunc)

    c_scheduler = optim.lr_scheduler.LambdaLR(model.optim_c, lr_lambda=lfunc)

    for epoch in range(config.experiment.epochs):
        model.train()
        g_ctr = 0
        d_losses = []
        for batch in fulldata.train_loader:
            g_ctr += 1
            dt, context, w = batch
            dt = dt.to(config.utils.device)
            context = context.to(config.utils.device)
            w = w.to(config.utils.device)

            # train critic
            model.optim_c.zero_grad()
            model.optim_g.zero_grad()
            d_loss = cramer_C_loss(model.G, model.C, dt, w, context, config)
            d_loss.backward()
            if config.experiment.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.C.parameters(), config.experiment.grad_clip)
            model.optim_c.step()
            d_losses.append(d_loss.item())

            # train generator
            if g_ctr == g_update_freq:

                model.optim_g.zero_grad()
                g_loss = cramer_G_loss(model.G, model.C, dt, w, context, config)
                g_loss.backward()
                if config.experiment.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.G.parameters(), config.experiment.grad_clip)
                model.optim_g.step()
                wandb.log({'Generator Loss': g_loss.item()})
                wandb.log({'Critic Loss': np.mean(d_losses)})
                d_losses = []
                g_ctr = 0
        print('epoch', epoch, 'done!')
        if (epoch + 1) % 10 == 0:
            c_scheduler.step()
            g_scheduler.step()
        if (epoch + 1) % 30 == 0:
            weights_name = 'bn_ln_cramerGAN_5layers' + str(epoch) + '.pt'
            model.save(weights_name)
            wandb.save(weights_name)
            model.eval()
            X = torch.tensor([])
            teacher_fake = torch.tensor([])
            context_ftrs = torch.tensor([])
            wghts = torch.tensor([])
            for batch in fulldata.train_loader:
                dt, context, w = batch
                dt = dt.to(config.utils.device)
                context = context.to(config.utils.device)
                w = w.to(config.utils.device)
                tfake = model.generate(dt, context)
                teacher_fake = torch.cat([teacher_fake, tfake.cpu()])
                X = torch.cat([X, dt.cpu()])
                context_ftrs = torch.cat([context_ftrs, context.cpu()])
                wghts = torch.cat([wghts, w.cpu()])

            clear_output(wait=True)
            log_histograms(X, teacher_fake, epoch)
            X = X.numpy()
            teacher_fake = teacher_fake.numpy()
            context_ftrs = context_ftrs.numpy()
            wghts = wghts.numpy()
            X = pd.DataFrame(data=X, columns=['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt'])
            teacher_fake = pd.DataFrame(data=teacher_fake,
                                        columns=['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt'])
            context_ftrs = pd.DataFrame(data=context_ftrs, columns=['Brunel_P', 'Brunel_ETA', 'nTracks_Brunel'])
            res = calculate_rocauc(config, context_ftrs, X, teacher_fake, wghts)
            wandb.log({'weighted ROC-AUC': res[0], 'unweighted ROC-AUC': res[1]})
            print('epoch', epoch, 'done!')