import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from tqdm import tqdm
import wandb

from utils import load_npz, to_graph, get_metadata, rollout
from gnn_network import LearnedSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

sweep_configuration = {
    "name": "gnn_sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "avg_loss"},
    "parameters": {
        "lr": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [4, 8, 16]},
        "optimizer": {"values": ["adam", "sgd"]},
        "noise_std": {"min": 0.0001, "max": 0.001},
        "window_size": {"min": 3, "max": 9},
    },
}


class OneStepDataset(pyg.data.Dataset):
    def __init__(self, path, metadata, window_size=5, noise_std=0.0):
        super().__init__()
        self.data = load_npz(path)
        self.metadata = metadata
        self.noise_std = noise_std

        self.window_size = window_size
        self.windows = []
        self._create_windows()

    def _create_windows(self):
        for sim in self.data[0].tolist().values():
            timesteps = sim[0]
            for i in range(0, len(timesteps) - self.window_size):
                trajectories = timesteps[i : i + self.window_size]
                assert len(trajectories) == self.window_size
                window = {
                    "size": len(sim[1]),
                    "particle_type": sim[1],
                    "trajectories": trajectories,
                }
                self.windows.append(window)

    def len(self):
        return len(self.windows)

    def get(self, idx):
        window = self.windows[idx]
        particle_type = torch.from_numpy(window["particle_type"])
        trajectories = window["trajectories"]
        target_position = np.array(trajectories[-1])
        position_seq = np.array(trajectories[:-1])
        target_position = torch.from_numpy(target_position)
        position_seq = torch.from_numpy(position_seq)

        with torch.no_grad():
            graph = to_graph(
                particle_type,
                position_seq,
                target_position,
                self.metadata,
                self.noise_std,
            )
        return graph


def rolloutMSE(
    simulator: torch.nn.Module,
    valid_simulation: pyg.loader.DataLoader,
    metadata: dict,
    # noise:
):
    total_loss = 0.0
    batch_count = 0
    simulator.eval()

    with torch.no_grad():
        for window in valid_simulation:
            rollout_out = rollout(simulator, window, metadata)
            rollout_out = rollout_out.permute(1, 0, 2)
            loss = ((rollout_out - window["position"]) ** 2).sum(dim=-1).mean()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count


def oneStepMSE(
    simulator: torch.nn.Module,
    valid_simulation: pyg.loader.DataLoader,
    metadata: dict,
) -> tuple:
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0

    scale = (
        torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2)
        + torch.tensor(metadata["acc_mean"])
    ).to(device)
    simulator.eval()
    with torch.no_grad():
        for window in valid_simulation:
            window = window.to(device)
            preds = simulator(window)

            mse = (((preds - window.y) * scale) ** 2).sum(dim=-1).mean()
            loss = ((preds - window.y) ** 2).mean()

            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1

    return total_loss / batch_count, total_mse / batch_count


def sweep_train(
    params,
    simulator: torch.nn.Module,
    train_loader: pyg.data.DataLoader,
    valid_loader: pyg.data.DataLoader,
    metadata: dict,
    epochs: int = 1,
    load_epoch: int = 0,
    sweep: bool = False,
) -> None:

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / 5e6)
    )

    # log the loss
    total_step = 0

    for i in range(epochs):
        simulator.train()
        total_loss = 0
        batch_count = 0

        # Only train on the first 1000 batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        for batch in progress_bar:
            if total_step > 1000:
                break
            optimizer.zero_grad()

            batch = batch.to(device)
            preds = simulator(batch)

            loss = loss_fn(preds, batch.y)
            loss.backward()

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "avg_loss": total_loss / batch_count,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if sweep:
                wandb.log(
                    {
                        "avg_loss": total_loss / batch_count,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            total_step += 1

            if total_step % 1000 == 0:
                simulator.eval()
                eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, metadata)
                tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                simulator.train()

    return


def start_train_sweep():
    wandb.init(project="gnn_project")
    params = wandb.config

    metadata = get_metadata()

    train_dataset = OneStepDataset(
        "data/processed/train.npz",
        metadata,
        window_size=params.window_size,
        noise_std=params.noise_std,
    )
    valid_dataset = OneStepDataset(
        "data/processed/valid.npz",
        metadata,
        window_size=params.window_size,
        noise_std=0.0,
    )
    train_loader = pyg.loader.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    valid_loader = pyg.loader.DataLoader(
        valid_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    print(f"Using device: {device}")

    simulator = LearnedSimulator(window_size=params.window_size)
    simulator = simulator.to(device)

    sweep_train(
        params,
        simulator,
        train_loader,
        valid_loader,
        metadata,
        epochs=1,
        load_epoch=0,
        sweep=True,
    )


def train(
    params: dict,
    simulator: torch.nn.Module,
    train_loader: pyg.data.DataLoader,
    valid_loader: pyg.data.DataLoader,
    metadata: dict,
) -> None:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / 5e6)
    )

    # log the loss
    total_step = 0

    for i in range(params["epochs"]):
        simulator.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        for batch in progress_bar:
            optimizer.zero_grad()

            batch = batch.cuda()
            preds = simulator(batch)

            loss = loss_fn(preds, batch.y)
            loss.backward()

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "avg_loss": total_loss / batch_count,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            total_step += 1

            if total_step % params["eval_interval"] == 0:
                simulator.eval()
                eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, metadata)
                tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                simulator.train()

            if total_step % params["rollout_interval"] == 0:
                simulator.eval()
                rollout_loss = rolloutMSE(simulator, valid_loader, metadata)
                tqdm.write(f"\nRollout Loss: {rollout_loss}")
                simulator.train()

            if total_step % params["save_interval"] == 0:
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(
                        params["model_path"],
                        f"epoch_{params["load_epoch"] + 1 + i}",
                        f"checkpoint_{total_step}.pt",
                    ),
                )
        torch.save(
            {
                "model": simulator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(
                params["model_path"],
                f"epoch_{params["load_epoch"] + 1 + i}",
                f"final_checkpoint_epoch_{params["load_epoch"] + 1 + i}.pt",
            ),
        )

    return


def default_train():
    params = {
        "epochs": 5,
        "batch_size": 8,
        "lr": 0.00305080362891173,
        "noise_std": 0.0008645544026624157,
        "save_interval": 1000,
        "eval_interval": 1000,
        "rollout_interval": 1000,
        "model_path": "checkpoints",
        "load_epoch": 0,
        "window_size": 3,
    }

    if not os.path.exists(params["model_path"]):
        os.makedirs(params["model_path"])

    for i in range(params["epochs"]):
        if not os.path.exists(f"checkpoints/epoch_{params["load_epoch"] + 1 + i}"):
            os.makedirs(f"checkpoints/epoch_{params["load_epoch"] + 1 + i}")

    metadata = get_metadata()

    train_dataset = OneStepDataset(
        "data/processed/train.npz",
        metadata,
        window_size=params["window_size"],
        noise_std=params["noise_std"],
    )
    valid_dataset = OneStepDataset(
        "data/processed/valid.npz",
        metadata,
        window_size=params["window_size"],
        noise_std=0.0,
    )
    train_loader = pyg.loader.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    valid_loader = pyg.loader.DataLoader(
        valid_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    simulator = LearnedSimulator(window_size=params["window_size"])
    simulator = simulator.cuda()

    # Load the model
    if params["load_epoch"] != 0:
        checkpoint = torch.load(
            f"checkpoints/epoch_{params["load_epoch"]}/final_checkpoint_epoch_{params["load_epoch"]}.pt",
            weights_only=True,
        )
        simulator.load_state_dict(checkpoint["model"])

    train(params, simulator, train_loader, valid_loader, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=bool, default=False)
    args = parser.parse_args()
    if args.sweep:
        sweep_id = wandb.sweep(sweep_configuration, project="second_sweep")
        wandb.agent(sweep_id, function=start_train_sweep)
    else:
        default_train()
