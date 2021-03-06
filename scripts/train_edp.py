import sys

from pyvacy import optim, analysis, sampling

from train import *

import model_tiny

def main(
    fpath, batch_size, noise_size,
    hidden_size, epochs, n_critic,
    n_generator,
    D_lr, G_lr,
    epoch_sample_cycle,
    epoch_sample_count,
    save_path,
    noise_multiplier,
    l2_norm_clip,
    weight_decay,
    tiny
):

    global model
    if tiny:
        model = model_tiny

    epsilon = analysis.epsilon(
        N=DATASET_SIZE,
        batch_size=batch_size,
        iterations=epochs,
        noise_multiplier=noise_multiplier
    )
    print("Epsilon: %.4f" % epsilon)

    tset = create_tensorset(fpath, MAX_TRAJ_LENGTH)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    modo_path = os.path.join(save_path, "I%05d.pkl")
    save_path = os.path.join(save_path, "I%05d.png")

    device = ["cpu", "cuda"][torch.cuda.is_available()]
    print("Using: %s" % device)

    G = model.Generator(noise_size, hidden_size, MAX_TRAJ_LENGTH).to(device)
    D = model.Discriminator(MAX_TRAJ_LENGTH, hidden_size).to(device)

    D_optim = optim.DPAdam(
        params=D.parameters(),
        lr=D_lr,
        betas=(0, 0.999),

        l2_norm_clip=l2_norm_clip,
        microbatch_size=1,
        minibatch_size=batch_size,
        noise_multiplier=noise_multiplier
    )
    G_optim = torch.optim.Adam(G.parameters(), lr=G_lr, betas=(0, 0.999))

    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        minibatch_size=batch_size,
        microbatch_size=1,
        iterations=epochs
    )

    for i, (X,) in enumerate(minibatch_loader(tset), 1):

        z = torch.randn(X.size(0), noise_size)

        D_optim.zero_grad()

        losses = []

        for Xi, zi in microbatch_loader(
            torch.utils.data.TensorDataset(X, z)
        ):
            Xi = Xi.to(device)
            zi = zi.to(device)

            with torch.no_grad():
                G.eval()
                Xh = G(zi)

            D.train()
            loss = D(Xh).mean() - D(Xi).mean()

            D_optim.zero_microbatch_grad()
            loss.backward()
            D_optim.microbatch_step()

            losses.append(loss.item())

        D_optim.step()

        print("[E%05d] %.4f" % (i, sum(losses)/len(losses)))

        if i % n_critic == 0:

            G.train()
            D.eval()

            for j in range(n_generator):
                z = torch.randn(batch_size, noise_size)
                G_optim.zero_grad()
                (-D(G(z.to(device)))).mean().backward()
                G_optim.step()

        if i % epoch_sample_cycle == 0:

            with torch.no_grad():
                G.eval()
                z = torch.randn(epoch_sample_count, noise_size).to(device)
                Xh = G(z)
                plot_and_save(Xh, save_path % i)

            modo = modo_path % i
            torch.save(G.state_dict(), modo)

            print("Saved %s" % modo)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", required=True)
    parser.add_argument("--epochs", type=int, default=20000)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--noise_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_critic", type=int, default=1)
    parser.add_argument("--n_generator", type=int, default=4)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--G_lr", type=float, default=5e-5)
    parser.add_argument("--epoch_sample_cycle", type=int, default=20)
    parser.add_argument("--epoch_sample_count", type=int, default=100)
    parser.add_argument("--save_path", required=True)

    parser.add_argument("--noise_multiplier", type=float, default=1.1)
    parser.add_argument("--l2_norm_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--tiny", type=int, required=True)
    args = parser.parse_args()

    main(**vars(args))