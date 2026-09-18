import matplotlib
matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from dataset import ColorDataset
from model import UNetColorization


def compute_psnr(img1, img2):

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100

    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):

    img1_gray = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    score, _ = ssim(img1_gray, img2_gray, full=True)

    return score


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ColorDataset("dataset_folder")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    model = UNetColorization().to(device)

    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    losses = []

    for epoch in range(100):

        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        for L, AB in loader:

            L = L.to(device)
            AB = AB.to(device)

            pred = model(L)

            loss = criterion(pred, AB)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pred_np = pred.detach().cpu().numpy()[0]
            gt_np = AB.detach().cpu().numpy()[0]
            L_np = L.detach().cpu().numpy()[0][0]

            pred_np = np.transpose(pred_np, (1,2,0))
            gt_np = np.transpose(gt_np, (1,2,0))

            L_img = (L_np * 255).astype(np.uint8)

            pred_AB = ((pred_np + 1) * 128).astype(np.uint8)
            gt_AB = ((gt_np + 1) * 128).astype(np.uint8)

            pred_lab = np.zeros((256,256,3), dtype=np.uint8)
            gt_lab = np.zeros((256,256,3), dtype=np.uint8)

            pred_lab[:,:,0] = L_img
            pred_lab[:,:,1:] = pred_AB

            gt_lab[:,:,0] = L_img
            gt_lab[:,:,1:] = gt_AB

            pred_bgr = cv2.cvtColor(pred_lab, cv2.COLOR_LAB2BGR)
            gt_bgr = cv2.cvtColor(gt_lab, cv2.COLOR_LAB2BGR)

            epoch_psnr += compute_psnr(pred_bgr, gt_bgr)
            epoch_ssim += compute_ssim(pred_bgr, gt_bgr)

        avg_loss = epoch_loss / len(loader)
        avg_psnr = epoch_psnr / len(loader)
        avg_ssim = epoch_ssim / len(loader)

        losses.append(avg_loss)

        print(
            f"Epoch {epoch+1}/100 | "
            f"Loss:{avg_loss:.4f} | "
            f"PSNR:{avg_psnr:.2f} | "
            f"SSIM:{avg_ssim:.3f}"
        )

    torch.save(model.state_dict(), "color_model.pth")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("training_loss.png")
    print("Loss graph saved as training_loss.png")


if __name__ == "__main__":
    main()