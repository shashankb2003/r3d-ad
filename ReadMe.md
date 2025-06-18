# Consistency Models for 3D Point Cloud Anomaly Detection

This repository contains the implementation of a **novel approach using Consistency Models (CMs)** for the task of **3D point cloud anomaly detection**. While 3D anomaly detection remains an underexplored area—especially for unstructured data like point clouds—this work builds upon and enhances prior methodologies such as **R3D-AD**, offering **faster inference and real-time applicability**.

---

## Motivation

3D anomaly detection plays a crucial role in domains like autonomous navigation, robotics, and industrial inspection. Traditional approaches struggle with the irregular and sparse nature of point clouds. A recent advancement, **R3D-AD**, leverages **Denoising Diffusion Probabilistic Models (DDPMs)** to reconstruct an anomaly-free version of an input point cloud. Comparing the reconstructed and original point clouds allows anomalies on the object’s surface to be detected.

However, DDPMs are **computationally expensive**, requiring **hundreds or thousands of sampling steps**. In contrast, **Consistency Models** offer **one-step generation**, making them ideal for real-time deployment.

---

## 🧠 Core Idea

Instead of a DDPM, our framework employs a **Consistency Model (CM)**, introduced by [Yang Song et al., 2023](https://arxiv.org/abs/2303.01469). These models are trained to produce consistent reconstructions across all diffusion timesteps, i.e., the output at timestep `t` should match the output at `t-1`. Recursively, the output at `t=0` becomes the ground truth.

We adopt **Consistency Training (CT)** over Consistency Distillation (CD), training the model from scratch using a **target model** and an **online model** in a setup inspired by reinforcement learning. The training objective is:
<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\color{white}\mathcal{L}^{N}_{\text{CD}}(\theta,\theta^{-};\phi):=\mathbb{E}\left[\lambda(t_n)\,d\left(f_{\theta}(\mathbf{x}_{t_{n&plus;1}},t_{n&plus;1}),f_{\theta^{-}}(\hat{\mathbf{x}}^{\phi}_{t_n},t_n)\right)\right]" title="\mathcal{L}^{N}_{\text{CD}}(\theta,\theta^{-};\phi):=\mathbb{E}\left[\lambda(t_n)\,d\left(f_{\theta}(\mathbf{x}_{t_{n+1}},t_{n+1}),f_{\theta^{-}}(\hat{\mathbf{x}}^{\phi}_{t_n},t_n)\right)\right]"/>
</p>


Where:
- `f_θ`: online model  
- `f_θ⁻`: target model (EMA of `F_θ`)  
- `d`: a distance metric (e.g., Chamfer Distance or EMD)  
- `λ`: a weighting coefficient

---

## Applications

Our primary use-case is **terrain anomaly detection via drone-mounted LiDAR**, where **fast and accurate reconstruction** is essential. The lightweight, single-step sampling ability of CMs makes them highly suitable for **real-time, on-device inference**.

---

## 🔧 Architecture Overview

- **Input**: 3D point cloud with surface anomalies synthesized using Patch-Gen
- **Encoder**: PointNet/PointNet++ backbone to encode into latent space  
- **CM**: Consistency Model trained using self-supervised consistency training  
- **Output**: Reconstructed (anomaly-free) point cloud  
- **Detection**: Chamfer/EMD-based anomaly scoring via input–output deviation

---

## Results

> _Sample visualizations of reconstructed vs input point clouds and anomaly maps will be added below._


---

## Citations
@article{song2023consistency,<br/>
  title={Consistency Models},<br/>
  author={Song, Yang and Meng, Chenlin and Ermon, Stefano},<br/>
  journal={arXiv preprint arXiv:2303.01469},<br/>
  year={2023}<br/>
}

@inproceedings{cao2023r3dad,
  title={R3D-AD: Reconstructing 3D Shapes for Unsupervised Anomaly Detection in Point Clouds},<br/>
  author={Cao, Xiyang and Zhang, Ziyang and Liu, Lanqing and Yan, Xiaokang and Yang, Kailun and Zhao<br/> Hengshuang and Geiger, Andreas and Shi, Jianping},<br/>
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},<br/>
  year={2023}<br/>
}
