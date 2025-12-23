import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor

# -----------------------------
# Config
# -----------------------------
IMG_ROOT = "/home/star/zzb/Pi3/assets/Ex4DGS"  # 新的图片根目录（包含 cam*/ 子目录）
DEBUG_BATCH = os.path.join("debug", "batch_data.pth")  # torch.save(batch_data, 'debug/batch_data.pth')
SAVE_DIR = os.path.join("debug", "images")
SAVE_NAME = "traj_from_batch_grid.png"
VIEWS_TO_USE = 4           # 可视化前 4 个视角
FRAMES_TO_DRAW = 6        # 每个视角挑选 10 帧
TARGET_W, TARGET_H = 504, 378  # 目标分辨率（W,H）
POINT_RADIUS = 2
POINT_COLOR = (255, 50, 50, 255)  # RGBA 红色


# -----------------------------
# Utils
# -----------------------------

def list_cam_dirs(img_root: str) -> List[str]:
    cam_dirs = sorted(glob.glob(os.path.join(img_root, "cam*")))
    if len(cam_dirs) == 0:
        raise FileNotFoundError(f"未在 {img_root} 下找到 cam* 目录")
    return cam_dirs[:VIEWS_TO_USE]


def get_image_path_for_time(cam_dir: str, time_idx: int) -> str:
    """优先尝试 6 位数命名，再回退 3 位数；若都不存在返回空串。"""
    p6 = os.path.join(cam_dir, f"{int(time_idx):06d}.png")
    if os.path.exists(p6):
        return p6
    p3 = os.path.join(cam_dir, f"{int(time_idx):03d}.png")
    if os.path.exists(p3):
        return p3
    return ""


def pick_frame_indices(num_frames: int, k: int) -> List[int]:
    if num_frames <= 0:
        return []
    k = min(max(1, k), num_frames)
    idx = np.linspace(0, num_frames - 1, k, dtype=int)
    uniq = np.unique(idx)
    out = list(uniq)
    while len(out) < k:
        out.append(out[-1])
    return out[:k]


def draw_points_on_pil(img_pil: Image.Image, xs: np.ndarray, ys: np.ndarray,
                        color: Tuple[int, int, int, int] = POINT_COLOR, radius: int = POINT_RADIUS):
    draw = ImageDraw.Draw(img_pil, "RGBA")
    W, H = img_pil.size
    for x, y in zip(xs, ys):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < W and 0 <= yi < H:
            draw.ellipse((xi - radius, yi - radius, xi + radius, yi + radius), fill=color)


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.isdir(IMG_ROOT):
        raise FileNotFoundError(f"图片根目录不存在: {IMG_ROOT}")
    cam_dirs = list_cam_dirs(IMG_ROOT)  # [cam00, cam06, cam11, cam16, ...]

    if not os.path.exists(DEBUG_BATCH):
        raise FileNotFoundError(f"未找到调试 batch: {DEBUG_BATCH}")
    pack = torch.load(DEBUG_BATCH, map_location="cpu")  # 这是 prepare_batch 的输出

    # 取关键数据
    keypoints_2d = pack.get('keypoints_2d', None)  # 期望 [T,V,K,2]
    time_ids = pack.get('time_ids', None)          # 期望 [T]

    if keypoints_2d is None:
        raise KeyError("batch_data 中缺少 keypoints_2d（[T,V,K,2]），无法进行轨迹可视化")
    if isinstance(keypoints_2d, np.ndarray):
        keypoints_2d = torch.from_numpy(keypoints_2d)
    if isinstance(time_ids, np.ndarray):
        time_ids = torch.from_numpy(time_ids)

    keypoints_2d = keypoints_2d.float().cpu()  # [T,V,K,2]
    T, V, K = keypoints_2d.shape[0], keypoints_2d.shape[1], keypoints_2d.shape[2]
    V_show = min(VIEWS_TO_USE, V, len(cam_dirs))

    if time_ids is None:
        # 若没有 time_ids，则按 t 索引直接取第 t 张图（基于文件排序）；
        # 但推荐在外部保存时保留 time_ids 以避免错位
        time_ids = torch.arange(T, dtype=torch.long)
    else:
        time_ids = time_ids.long().cpu()

    # 均匀挑选 10 个时间步索引（按 T 的下标），并取对应的 time_id 映射到图片文件名
    t_indices = pick_frame_indices(T, FRAMES_TO_DRAW)

    grid_images: List[torch.Tensor] = []

    for v in range(V_show):
        for ti in t_indices:
            t_img_id = int(time_ids[ti].item())
            cam_dir = cam_dirs[v]
            img_path = get_image_path_for_time(cam_dir, t_img_id)
            if img_path == "":
                # 退化：按排序索引取第 ti 张
                frames = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
                if len(frames) == 0:
                    raise FileNotFoundError(f"{cam_dir} 下没有 *.png")
                img_path = frames[min(ti, len(frames) - 1)]

            with Image.open(img_path) as img_pil:
                img_pil = img_pil.convert("RGB")
                if img_pil.size != (TARGET_W, TARGET_H):
                    img_pil = img_pil.resize((TARGET_W, TARGET_H), resample=Image.BILINEAR)

                # 当前视角 v、时间下标 ti 的关键点 [K,2]
                kps = keypoints_2d[ti, v]  # [K,2]
                if kps.numel() > 0:
                    xs = kps[:, 0].numpy()
                    ys = kps[:, 1].numpy()
                    draw_points_on_pil(img_pil, xs, ys, color=POINT_COLOR, radius=POINT_RADIUS)

                grid_images.append(to_tensor(img_pil))  # [3,H,W] in [0,1]

    if len(grid_images) == 0:
        raise RuntimeError("没有生成任何可视化图像，请检查 batch_data 中的 keypoints_2d / time_ids 和图像目录结构")

    grid_tensor = torch.stack(grid_images, dim=0)
    out_path = os.path.join(SAVE_DIR, SAVE_NAME)
    save_image(grid_tensor, out_path, nrow=FRAMES_TO_DRAW)
    print(f"已保存叠加轨迹大图到: {out_path}")


if __name__ == "__main__":
    main()
