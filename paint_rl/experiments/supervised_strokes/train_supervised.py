"""
Trains a model on the supervised stroke task.
"""
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import json

import wandb

from paint_rl.experiments.supervised_strokes.gen_supervised import (
    IMG_SIZE,
    NUM_IMAGES,
    gen_sample,
)


class StrokeNet(nn.Module):
    def __init__(self, size: int):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            # Input is num channels + num pos elements
            nn.Conv2d(3 + 2, 12, 3),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(12, 32, 3),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            # nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.ff = nn.Sequential(nn.Linear(128, 4))
        # Pos encoding
        w = torch.arange(0, size).unsqueeze(0).repeat([size, 1])
        h = torch.arange(0, size).unsqueeze(0).repeat([size, 1]).T
        self.pos = nn.Parameter(
            torch.stack([w, h]).unsqueeze(0) / (size / 2.0) - 1.0, requires_grad=False
        )

    def forward(self, x):
        batch_size = x.shape[0]
        pos_encoding = self.pos.repeat([batch_size, 1, 1, 1])
        x = torch.cat([x, pos_encoding], dim=1)
        x = self.net(x)
        x = torch.max(torch.max(x, 3).values, 2).values
        x = self.ff(x)
        return x


def main():
    # Load dataset
    img_size = IMG_SIZE
    ds_x = torch.zeros([NUM_IMAGES, 3, img_size, img_size], dtype=torch.int8)
    print("Loading data.")
    for i in tqdm(range(NUM_IMAGES)):
        img = Image.open(f"supervised/{i}.png")
        img_data = torch.swapaxes(torch.from_numpy(np.array(img)), 0, -1) / 255.0
        ds_x[i] = img_data
    with open("supervised_paths.json", "r") as f:
        paths = json.load(f)
    ds_y = (
        torch.tensor(
            [[item[1][0], item[1][1], item[2][0], item[2][1]] for item in paths]
        )
        / (img_size / 2)
        - 1.0
    )
    del paths
    train_x, valid_x, train_y, valid_y = train_test_split(ds_x, ds_y, train_size=0.8)
    del ds_x, ds_y
    train_size = train_x.shape[0]

    wandb.init(
        project="paint-rl",
        entity="bensgiacalone",
        config={
            "experiment": "supervised stroke rendering",
        },
    )

    # Train model
    net = StrokeNet(img_size).cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    batch_size = 512
    train_x = train_x.float().cuda()
    train_y = train_y.float().cuda()
    valid_batch_x = valid_x.float().cuda()
    valid_batch_y = valid_y.float().cuda()
    for j in tqdm(range(1000000)):
        mean_loss = 0.0
        num_batches = train_size // batch_size
        indices = torch.from_numpy(
            np.random.choice(train_size, train_size, replace=False)
        )
        for i in range(num_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            out = net(batch_x)
            loss = ((out - batch_y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss += loss.item()
        mean_loss /= num_batches

        # Evaluate
        with torch.no_grad():
            valid_out = net(valid_batch_x)
            valid_loss = ((valid_out - valid_batch_y) ** 2).mean()
            wandb.log({"loss": mean_loss, "valid_loss": valid_loss.item()})

        # Generate new data
        if (j + 1) % 30 == 0:
            samples = [gen_sample(IMG_SIZE) for _ in range(train_size)]
            imgs = [x[0] for x in samples]
            paths = [x[1] for x in samples]
            del samples
            train_x = (
                torch.stack(
                    [
                        torch.swapaxes(torch.from_numpy(np.array(img)), 0, -1) / 255.0
                        for img in imgs
                    ]
                )
                .float()
                .cuda()
            )
            train_y = (
                (
                    torch.tensor(
                        [
                            [item[1][0], item[1][1], item[2][0], item[2][1]]
                            for item in paths
                        ]
                    )
                    / (img_size / 2)
                    - 1.0
                )
                .float()
                .cuda()
            )

        # Save
        if (j + 1) % 20 == 0:
            torch.save(net.state_dict(), "temp/stroke_net.pt")


if __name__ == "__main__":
    main()
