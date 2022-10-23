
# Batch3DMOT
[**arXiv**](https://arxiv.org/abs/XXXX.XXXXX) | [**website**](http://batch3dmot.cs.uni-freiburg.de/)

<p align="center">
  <img src="video_banner.gif" alt="Overview of Batch3DMOT architecture" width="850" />
</p>

This repository is the official implementation of the paper:

> **3D Multi-Object Tracking Using Graph Neural Networks with Cross-Edge Modality Attention**
>
> [Martin Büchner](https://rl.uni-freiburg.de/people/buechner)  and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada).
>
> *arXiv preprint arXiv:XXXX.XXXXX, 2022*

<p align="center">
  <img src="batch3dmot_architecture.png" alt="Overview of Batch3DMOT architecture" width="850" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{buechner2022batch3dmot,
	  title={3D Multi-Object Tracking Using Graph Neural Networks with Cross-Edge Modality Attention},
	  author={B{\"u}chner, Martin and Valada, Abhinav},
	  journal={arXiv preprint arXiv:XXXX.XXXX},
	  year={2022}
}
```

## 📔 Abstract

Online 3D multi-object tracking (MOT) has witnessed significant research interest in recent years, largely driven by demand from the autonomous systems community. However, 3D offline MOT is relatively less explored. Labeling 3D trajectory scene data at a large scale while not relying on high-cost human experts is still an open research question. In this work, we propose Batch3DMOT that follows the tracking-by-detection paradigm and represents real-world scenes as directed, acyclic, and category-disjoint tracking graphs that are attributed using various modalities such as camera, LiDAR, and radar. We present a multi-modal graph neural network that uses a cross-edge attention mechanism mitigating modality intermittence, which translates into sparsity in the graph domain. Additionally, we present attention-weighted convolutions over frame-wise k-NN neighborhoods as suitable means to allow information exchange across disconnected graph components. We evaluate our approach using various sensor modalities and model configurations on the challenging nuScenes and KITTI datasets. Extensive experiments demonstrate that our proposed approach yields an overall improvement of 2.8% in the AMOTA score on nuScenes thereby setting a new benchmark for 3D tracking methods and successfully enhances false positive filtering.

## 👨‍💻 Code Release

### Installation
- Download nuScenes dataset from [here](https://www.nuscenes.org/download).
- Download Megvii and CenterPoint detections. You may use `src/utils/concat_jsons.py` to obtain mini-split results.
- Define relevant paths in `*_config.yaml`
  * The `tmp`-folder holds preprocessed graph data while the `data`-folder holds the raw nuScenes dataset.
  * Adjust package paths to match your local setup.
- Generate 2D image annotation by running `python nuscenes/scripts/export_2d_annotations_as_json.py --dataroot=/path/to/nuscdata --version=v1.0-trainval` and place it under the nuScenes data directory.

### Preprocessing
- Generate metadata and GT for feature encoder training:
  * `python batch_3dmot/preprocessing/preprocess_img.py --config cl_config.yaml`
  * `python batch_3dmot/preprocessing/preprocess_lidar.py --config cl_config.yaml`, 
  * `python batch_3dmot/preprocessing/preprocess_radar.py  --config cl_config.yaml` 
- Train feature encoders:
  * `python batch_3dmot/preprocessing/train_resnet_ae.py --config cl_config.yaml`
  * `python batch_3dmot/preprocessing/train_pointnet.py --config cl_config.yaml`
  * `python batch_3dmot/preprocessing/train_radarnet.py --config cl_config.yaml`
- Construct disjoint, directed tracking graphs either using modalities or not:
  * `python batch_3dmot/preprocessing/construct_detection_graphs_disjoint_parallel.py --config cl_config.yaml`
  * `python batch_3dmot/preprocessing/construct_detection_graphs_disjoint_parallel_only_poses.py --config pose_config.yaml`


### Training and Evaluation
- Train Batch3DMOT (poses-only or using modalities):
  * `python batch_3dmot/train_poses_only.py --config pose_config.yaml`
  * `python batch_3dmot/wandb_train.py --config cl_config.yaml`
- Perform inference using trained model (poses-only, pose+camera or using more modalities):
  * `python batch_3dmot/predict_detections_poses.py --config pose_config.yaml`
  * `python batch_3dmot/predict_detctions_img.py --config cl_config.yaml`
  * `python batch_3dmot/predict_detections.py --config cl_config.yaml`
- Evaluate produced tracking result:
  * `python batch_3dmot/eval/eval_nuscenes.py --config ***_config.yaml`

## 👩‍⚖️ License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.
