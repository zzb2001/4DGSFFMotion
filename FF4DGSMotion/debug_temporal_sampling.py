"""
时间感知采样调试工具

用于分析和可视化 prepare_canonical 的采样过程
"""

import torch
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class TemporalSamplingDebugger:
    """时间感知采样调试器"""
    
    def __init__(self, model, output_dir: str = './debug_output'):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {}
    
    def analyze_input_points(self, points_3d: torch.Tensor) -> Dict:
        """分析输入点云的统计信息"""
        T, N, _ = points_3d.shape
        device = points_3d.device
        
        stats = {
            'total_points': T * N,
            'frames': T,
            'points_per_frame': N,
            'valid_points_per_frame': [],
            'point_ranges': [],
            'motion_magnitude': [],
        }
        
        for t in range(T):
            pts_t = points_3d[t]
            valid_mask = torch.isfinite(pts_t).all(dim=-1)
            valid_count = valid_mask.sum().item()
            stats['valid_points_per_frame'].append(valid_count)
            
            # 点云范围
            pts_valid = pts_t[valid_mask]
            if valid_count > 0:
                min_pt = pts_valid.min(dim=0)[0]
                max_pt = pts_valid.max(dim=0)[0]
                range_pt = (max_pt - min_pt).norm().item()
                stats['point_ranges'].append(range_pt)
        
        # 运动幅度
        for t in range(T - 1):
            pts_t = points_3d[t]
            pts_t1 = points_3d[t + 1]
            
            valid_mask_t = torch.isfinite(pts_t).all(dim=-1)
            valid_mask_t1 = torch.isfinite(pts_t1).all(dim=-1)
            
            if valid_mask_t.sum() > 0 and valid_mask_t1.sum() > 0:
                center_t = pts_t[valid_mask_t].mean(dim=0)
                center_t1 = pts_t1[valid_mask_t1].mean(dim=0)
                motion_mag = (center_t1 - center_t).norm().item()
                stats['motion_magnitude'].append(motion_mag)
        
        self.stats['input'] = stats
        return stats
    
    def print_input_stats(self):
        """打印输入统计信息"""
        if 'input' not in self.stats:
            print("Please call analyze_input_points first")
            return
        
        stats = self.stats['input']
        print("\n" + "="*60)
        print("INPUT POINT CLOUD STATISTICS")
        print("="*60)
        print(f"Total points: {stats['total_points']:,}")
        print(f"Frames: {stats['frames']}")
        print(f"Points per frame: {stats['points_per_frame']:,}")
        print(f"\nValid points per frame:")
        for t, count in enumerate(stats['valid_points_per_frame']):
            print(f"  Frame {t}: {count:,}")
        
        print(f"\nPoint cloud range (per frame):")
        for t, range_val in enumerate(stats['point_ranges']):
            print(f"  Frame {t}: {range_val:.3f}m")
        
        if stats['motion_magnitude']:
            print(f"\nMotion magnitude (frame-to-frame):")
            for t, mag in enumerate(stats['motion_magnitude']):
                print(f"  Frame {t} → {t+1}: {mag:.4f}m")
            print(f"  Average: {np.mean(stats['motion_magnitude']):.4f}m")
            print(f"  Max: {np.max(stats['motion_magnitude']):.4f}m")
    
    def analyze_sampling_process(self, points_3d: torch.Tensor, 
                                 k_per_frame: int = 2000,
                                 voxel_size: float = 0.01) -> Dict:
        """分析采样过程的每一步"""
        T, N, _ = points_3d.shape
        device = points_3d.device
        dtype = points_3d.dtype
        
        stats = {
            'k_per_frame': k_per_frame,
            'voxel_size': voxel_size,
            'per_frame_samples': [],
            'total_after_sampling': 0,
            'voxel_dedup_stats': {},
            'time_stability_dist': [],
        }
        
        # Step 1: 分帧采样
        points_sampled_list = []
        frame_indices = []
        
        for t in range(T):
            pts_t = points_3d[t]
            valid_mask = torch.isfinite(pts_t).all(dim=-1)
            pts_valid = pts_t[valid_mask]
            
            if pts_valid.shape[0] > k_per_frame:
                idx = torch.randperm(pts_valid.shape[0], device=device)[:k_per_frame]
                pts_sampled = pts_valid[idx]
                sample_count = k_per_frame
            else:
                pts_sampled = pts_valid
                sample_count = pts_valid.shape[0]
            
            points_sampled_list.append(pts_sampled)
            frame_indices.append(torch.full((pts_sampled.shape[0],), t, dtype=torch.long, device=device))
            stats['per_frame_samples'].append(sample_count)
        
        points_all = torch.cat(points_sampled_list, dim=0)
        frame_ids = torch.cat(frame_indices, dim=0)
        stats['total_after_sampling'] = points_all.shape[0]
        
        # Step 2: 去重合并
        voxel_indices = torch.floor(points_all / voxel_size).long()
        unique_voxels, inverse_indices = torch.unique(
            voxel_indices, dim=0, return_inverse=True
        )
        
        stats['voxel_dedup_stats']['num_voxels'] = len(unique_voxels)
        stats['voxel_dedup_stats']['compression_ratio'] = (
            points_all.shape[0] / len(unique_voxels)
        )
        
        # 分析每个 voxel 中的点数和时间覆盖
        points_per_voxel = []
        frames_per_voxel = []
        
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            pts_in_voxel = points_all[mask]
            frames_in_voxel = frame_ids[mask]
            
            points_per_voxel.append(mask.sum().item())
            num_frames = len(torch.unique(frames_in_voxel))
            frames_per_voxel.append(num_frames)
            
            # 时间稳定性
            stability = num_frames / T
            stats['time_stability_dist'].append(stability)
        
        stats['voxel_dedup_stats']['avg_points_per_voxel'] = np.mean(points_per_voxel)
        stats['voxel_dedup_stats']['avg_frames_per_voxel'] = np.mean(frames_per_voxel)
        stats['voxel_dedup_stats']['max_points_per_voxel'] = np.max(points_per_voxel)
        stats['voxel_dedup_stats']['max_frames_per_voxel'] = np.max(frames_per_voxel)
        
        self.stats['sampling'] = stats
        return stats
    
    def print_sampling_stats(self):
        """打印采样统计信息"""
        if 'sampling' not in self.stats:
            print("Please call analyze_sampling_process first")
            return
        
        stats = self.stats['sampling']
        print("\n" + "="*60)
        print("SAMPLING PROCESS STATISTICS")
        print("="*60)
        print(f"Parameters:")
        print(f"  k_per_frame: {stats['k_per_frame']}")
        print(f"  voxel_size: {stats['voxel_size']}")
        
        print(f"\nStep 1: Per-frame sampling")
        total_sampled = sum(stats['per_frame_samples'])
        print(f"  Total sampled: {total_sampled:,} points")
        print(f"  Per-frame breakdown:")
        for t, count in enumerate(stats['per_frame_samples']):
            print(f"    Frame {t}: {count:,}")
        
        voxel_stats = stats['voxel_dedup_stats']
        print(f"\nStep 2: Voxel deduplication")
        print(f"  Unique voxels: {voxel_stats['num_voxels']:,}")
        print(f"  Compression ratio: {voxel_stats['compression_ratio']:.2f}x")
        print(f"  Avg points per voxel: {voxel_stats['avg_points_per_voxel']:.2f}")
        print(f"  Max points per voxel: {voxel_stats['max_points_per_voxel']}")
        
        print(f"\nStep 3: Time stability analysis")
        print(f"  Avg frames per voxel: {voxel_stats['avg_frames_per_voxel']:.2f}")
        print(f"  Max frames per voxel: {voxel_stats['max_frames_per_voxel']}")
        
        time_stab = np.array(stats['time_stability_dist'])
        print(f"  Time stability distribution:")
        print(f"    Mean: {time_stab.mean():.4f}")
        print(f"    Std: {time_stab.std():.4f}")
        print(f"    Min: {time_stab.min():.4f}, Max: {time_stab.max():.4f}")
        print(f"    Percentiles: 25%={np.percentile(time_stab, 25):.4f}, "
              f"50%={np.percentile(time_stab, 50):.4f}, "
              f"75%={np.percentile(time_stab, 75):.4f}")
    
    def analyze_final_gaussians(self) -> Dict:
        """分析最终高斯的统计信息"""
        if not self.model._world_cache.get('prepared'):
            print("Model not prepared yet")
            return {}
        
        mu = self.model._world_cache['surfel_mu']
        normal = self.model._world_cache['surfel_normal']
        radius = self.model._world_cache['surfel_radius']
        conf = self.model._world_cache['surfel_confidence']
        
        stats = {
            'num_gaussians': mu.shape[0],
            'mu_range': (mu.min(dim=0)[0], mu.max(dim=0)[0]),
            'radius_stats': {
                'mean': radius.mean().item(),
                'std': radius.std().item(),
                'min': radius.min().item(),
                'max': radius.max().item(),
            },
            'confidence_stats': {
                'mean': conf.mean().item(),
                'std': conf.std().item(),
                'min': conf.min().item(),
                'max': conf.max().item(),
                'percentiles': {
                    '25': conf.quantile(0.25).item(),
                    '50': conf.quantile(0.5).item(),
                    '75': conf.quantile(0.75).item(),
                }
            }
        }
        
        self.stats['final_gaussians'] = stats
        return stats
    
    def print_final_stats(self):
        """打印最终高斯统计信息"""
        if 'final_gaussians' not in self.stats:
            self.analyze_final_gaussians()
        
        stats = self.stats['final_gaussians']
        print("\n" + "="*60)
        print("FINAL GAUSSIANS STATISTICS")
        print("="*60)
        print(f"Number of Gaussians: {stats['num_gaussians']:,}")
        
        mu_min, mu_max = stats['mu_range']
        print(f"\nGaussian center range:")
        print(f"  X: [{mu_min[0]:.4f}, {mu_max[0]:.4f}]")
        print(f"  Y: [{mu_min[1]:.4f}, {mu_max[1]:.4f}]")
        print(f"  Z: [{mu_min[2]:.4f}, {mu_max[2]:.4f}]")
        
        r_stats = stats['radius_stats']
        print(f"\nRadius statistics:")
        print(f"  Mean: {r_stats['mean']:.6f}")
        print(f"  Std: {r_stats['std']:.6f}")
        print(f"  Range: [{r_stats['min']:.6f}, {r_stats['max']:.6f}]")
        
        c_stats = stats['confidence_stats']
        print(f"\nConfidence statistics:")
        print(f"  Mean: {c_stats['mean']:.4f}")
        print(f"  Std: {c_stats['std']:.4f}")
        print(f"  Range: [{c_stats['min']:.4f}, {c_stats['max']:.4f}]")
        print(f"  Percentiles:")
        for p, v in c_stats['percentiles'].items():
            print(f"    {p}%: {v:.4f}")
    
    def plot_confidence_distribution(self, save_path: str = None):
        """绘制置信度分布"""
        if not self.model._world_cache.get('prepared'):
            print("Model not prepared yet")
            return
        
        conf = self.model._world_cache['surfel_confidence'].cpu().numpy().flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 直方图
        axes[0].hist(conf, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(conf.mean(), color='r', linestyle='--', label=f'Mean: {conf.mean():.4f}')
        axes[0].axvline(np.median(conf), color='g', linestyle='--', label=f'Median: {np.median(conf):.4f}')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Confidence Distribution (Histogram)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_conf = np.sort(conf)
        axes[1].plot(sorted_conf, np.arange(len(sorted_conf)) / len(sorted_conf), linewidth=2)
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Confidence Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'confidence_distribution.png'
        
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_time_stability_distribution(self, save_path: str = None):
        """绘制时间稳定性分布"""
        if 'sampling' not in self.stats:
            print("Please call analyze_sampling_process first")
            return
        
        time_stab = np.array(self.stats['sampling']['time_stability_dist'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 直方图
        axes[0].hist(time_stab, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(time_stab.mean(), color='r', linestyle='--', label=f'Mean: {time_stab.mean():.4f}')
        axes[0].set_xlabel('Time Stability')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Time Stability Distribution (Histogram)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_ts = np.sort(time_stab)
        axes[1].plot(sorted_ts, np.arange(len(sorted_ts)) / len(sorted_ts), linewidth=2)
        axes[1].set_xlabel('Time Stability')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Time Stability Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'time_stability_distribution.png'
        
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
        plt.close()
    
    def generate_report(self, points_3d: torch.Tensor):
        """生成完整的调试报告"""
        print("\n" + "="*60)
        print("TEMPORAL SAMPLING DEBUG REPORT")
        print("="*60)
        
        # 分析输入
        self.analyze_input_points(points_3d)
        self.print_input_stats()
        
        # 分析采样过程
        self.analyze_sampling_process(points_3d)
        self.print_sampling_stats()
        
        # 分析最终结果
        self.analyze_final_gaussians()
        self.print_final_stats()
        
        # 绘制图表
        print("\nGenerating plots...")
        self.plot_confidence_distribution()
        self.plot_time_stability_distribution()
        
        print("\n" + "="*60)
        print("REPORT COMPLETE")
        print("="*60)


def main():
    """示例使用"""
    from FF4DGSMotion.models.FF4DGSMotion import Trellis4DGS4DCanonical
    
    # 创建模型
    model = Trellis4DGS4DCanonical().cuda()
    
    # 创建调试器
    debugger = TemporalSamplingDebugger(model)
    
    # 创建示例数据
    points_3d = torch.randn(6, 190512, 3).cuda()
    
    # 重置缓存
    model.reset_cache()
    
    # 准备 canonical
    model.prepare_canonical(points_3d)
    
    # 生成报告
    debugger.generate_report(points_3d)


if __name__ == '__main__':
    main()






