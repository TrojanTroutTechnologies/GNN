import os
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from tqdm import tqdm

from src.utils import OneStepDataset, get_metadata, rollout
from src.gnn import LearnedSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")


def rolloutMSE(
    simulator: torch.nn.Module,
    rollout_data: pyg.data.Dataset,
    metadata: dict,
):
    simulator.eval()
    with torch.no_grad():
        rollout_traj = rollout(simulator, rollout_data, metadata)
        ground_truth = torch.from_numpy(
            np.array(rollout_data.data[0].tolist()["simulation_0"][0])
        )
        return ((rollout_traj - ground_truth) ** 2).mean()


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


def train_loop(
    params: dict,
    simulator: torch.nn.Module,
    train_loader: pyg.data.DataLoader,
    valid_loader: pyg.data.DataLoader,
    rollout_data: pyg.data.Dataset,
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
                rollout_loss = rolloutMSE(simulator, rollout_data, metadata)
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


def start_training():
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

    train_loop(params, simulator, train_loader, valid_loader, valid_dataset, metadata)


if __name__ == "__main__":
    start_training()
