# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/cluster/scratch/takmaza/CVL/kitti_data')
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='/cluster/scratch/takmaza/CVL/kitti-animations')

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0]) #, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--video_save_frequency", type=int, help="video saving frequency", default=1)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=2) #batch_size=12
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options

        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--num_pairs", type=int, default = 100000)
        self.parser.add_argument("--d_lim", type=int, default = 30)

        # ABLATION options
        self.parser.add_argument("--save_depth_video", help="if set, saves depth prediction videos", action="store_true")
        self.parser.add_argument("--load_depth", help="uses the flow obtained from the raft network", action="store_true")
        self.parser.add_argument("--use_raft_flow", help="uses the flow obtained from the raft network", action="store_true")
        self.parser.add_argument("--loss_percentage", help="w*e/d", action="store_true")
        self.parser.add_argument("--check_both_weights", help="calculate gt weights for comparison", action="store_true")
        self.parser.add_argument("--use_embedding_weights", help="if set, uses the learned embeddings to derive the weights", action="store_true")
        self.parser.add_argument("--use_only_sparse_sup", help="if set, uses only sparse gt depth supervision", action="store_true")
        self.parser.add_argument("--use_sparse_order_sup", help="if set, uses sparse point order supervision", action="store_true" )
        self.parser.add_argument("--use_rel_dist_normalization", help="if set, uses relative distance normalization for the calculation of loss", action="store_true")
        self.parser.add_argument("--check_gt_weights", help="calculate gt weights for comparison", action="store_true")
        self.parser.add_argument("--loss_divisor_1", help="multiplies the loss by (1/s1 + 1/s2)", action="store_true")
        self.parser.add_argument("--loss_divisor_2", help="multiplies the loss by (1/(s1*s2))", action="store_true")
        self.parser.add_argument("--separate_enc_wo_flow", help="separates encoders wo opt flow", action="store_true")
        self.parser.add_argument("--experiment_one", help="separates weights from iso loss", action="store_true")
        self.parser.add_argument("--use_optical_flow_emb", help="if set, uses the weight learning network which uses optical flow and image luminance", action="store_true")
        self.parser.add_argument("--use_cyclic_cons", help="if set, uses an additional weight supervision loss based on the predicted depth", action="store_true")
        self.parser.add_argument("--scale_cyclic", help="if set, cyclic loss is scaled with the isometric loss", action="store_true")
        self.parser.add_argument("--ovf_to_weights", help="if set, uses only one frame to overfit to the gt weights", action="store_true")
        self.parser.add_argument("--use_frame_normalization", help="if set, performs normalization across frames", action="store_true")
        self.parser.add_argument("--use_smoothness_loss", help="if set, adds smoothness loss to the objective", action="store_true")
        self.parser.add_argument("--use_weight_sup", help="if set, uses gt weight supervision for ", action="store_true")
        self.parser.add_argument("--use_cosine_dist", help="if set, uses cosine dist for weight calculation", action="store_true")
        self.parser.add_argument("--force_w_01", help="if set, forces w to approach either 0 or 1", action="store_true")
        self.parser.add_argument("--learn_weights", help="if set, learns the pair weights", action="store_true")
        self.parser.add_argument("--scaled_weight_loss", help="if set, weight loss is scaled with the isometric loss", action="store_true")
        self.parser.add_argument("--save_video", help="if set, saves depth prediction videos", action="store_true")
        self.parser.add_argument("--save_embeddings", help="if set, saves pixel embeddings", action="store_true")
        self.parser.add_argument("--seq_random_pairs", help="if set, uses full sequences", action="store_true")
        self.parser.add_argument("--track_unsup_loss", help="if set, tracks the unsup loss as well", action="store_true")
        self.parser.add_argument("--use_clamping", help="if set, clamps the ground truth depth", action="store_true")
        self.parser.add_argument("--use_gt_depth", help="if set, uses ground truth depth data for the training", action="store_true")
        self.parser.add_argument("--use_gt_distances", help="if set, uses ground truth data for the calculation of pairwise distances", action="store_true")
        self.parser.add_argument("--use_gt_weights", help="if set, uses ground truth weights for the calculation of loss", action="store_true")
        self.parser.add_argument("--use_gt_scale", help="if set, uses ground truth scale for the calculation of loss", action="store_true")
        self.parser.add_argument("--use_01_norm", help="if set, normalizes the ground truth depth", action="store_true")
        self.parser.add_argument("--use_dist_normalization", help="if set, uses distance normalization for the calculation of loss", action="store_true")
        self.parser.add_argument("--v1_multiscale", help="if set, uses monodepth v1 multiscale", action="store_true")
        self.parser.add_argument("--avg_reprojection", help="if set, uses average reprojection loss", action="store_true")
        self.parser.add_argument("--disable_automasking", help="if set, doesn't do auto-masking", action="store_true")
        self.parser.add_argument("--predictive_mask", help="if set, uses a predictive masking scheme as in Zhou et al", action="store_true")
        self.parser.add_argument("--no_ssim", help="if set, disables ssim in the loss", action="store_true")
        self.parser.add_argument("--weights_init", type=str, help="pretrained or scratch", default="pretrained", choices=["pretrained", "scratch"])


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
