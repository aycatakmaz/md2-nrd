# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import time
import os
import os.path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
import pdb
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import animation

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

class Trainer:
    def __init__(self, options):
        #pdb.set_trace()
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        test_path = os.path.join(os.path.dirname(__file__), "splits", 'eigen_benchmark', "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        test_filenames = readlines(test_path.format("test"))

        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext, is_flow=True, device=self.device)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext, is_flow=True, device=self.device)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        vid_dataset_val = self.dataset(
            self.opt.data_path, sorted(val_filenames), self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext, is_flow=False, device=self.device)

        vid_dataset_test = self.dataset(
            self.opt.data_path, sorted(test_filenames), self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext, is_flow=False, device=self.device)

        self.vid_loader_val = DataLoader(
            vid_dataset_val, 1, False,
            num_workers=0, pin_memory=True, drop_last=True)
        self.vid_loader_test = DataLoader(
            vid_dataset_test, 1, False,
            num_workers=0, pin_memory=True, drop_last=True)

        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            if  (self.epoch+1)%self.opt.video_save_frequency==0: #(self.epoch+1)%5==0: #self.epoch==0: #
                print('printing epoch: ', self.epoch)
                self.run_epoch(log_vid=True)
            else:
                self.run_epoch(log_vid=False)


    def run_epoch(self,log_vid=False):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            if self.opt.save_video:
                if log_vid==True and batch_idx%1000==0:
                    print('saving depth video')
                    self.save_depth_video()
                    print('saved depth video')

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        eps = 1e-9
        import time
        
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        
        tgt_features = self.models["encoder"](inputs["color", 0, 0])
        _,depth_tgt_list = disp_to_depth(self.models["depth"](tgt_features)['disp', 0], self.opt.min_depth, self.opt.max_depth) #depth_rescale(self.models["depth"](tgt_features)['disp', 0],0, self.opt.clamp_max)
        #pdb.set_trace()
        depth_tgt_cam_points = self.backproject_depth[0](depth_tgt_list, inputs[("inv_K",0)])
        depth_tgt_cam_points = depth_tgt_cam_points[:,0:3,:].view(self.opt.batch_size,3,self.opt.height,self.opt.width)
        depth_tgt_cam_points = depth_tgt_cam_points.permute(0, 2, 3, 1)

        ref_features = self.models["encoder"](inputs["color", 1, 0])
        _,depth_ref_list = disp_to_depth(self.models["depth"](ref_features)['disp', 0], self.opt.min_depth, self.opt.max_depth) #depth_rescale(self.models["depth"](ref_features)['disp', 0],0, self.opt.clamp_max)

        depth_ref_cam_points = self.backproject_depth[0](depth_ref_list, inputs[("inv_K",0)])
        depth_ref_cam_points = depth_ref_cam_points[:,0:3,:].view(self.opt.batch_size,3,self.opt.height,self.opt.width)
        depth_ref_cam_points = depth_ref_cam_points.permute(0, 2, 3, 1)

        #pdb.set_trace()

        corresp_list = inputs['corresp'].type(torch.LongTensor)
        p1, p2 = inputs['pairs'][:,:,0:2], inputs['pairs'][:,:,2:]
        corresp_p1 = torch.cat([torch.unsqueeze(corresp_list[iii,p1[iii,:,0],p1[iii,:,1]],dim=0) for iii in range(corresp_list.shape[0])], dim=0)
        corresp_p2 = torch.cat([torch.unsqueeze(corresp_list[iii,p2[iii,:,0],p2[iii,:,1]],dim=0) for iii in range(corresp_list.shape[0])], dim=0)
        xyz_p1 = torch.cat([torch.unsqueeze(depth_tgt_cam_points[iii,p1[iii,:,0],p1[iii,:,1]],dim=0) for iii in range(depth_tgt_cam_points.shape[0])], dim=0)
        xyz_p2 = torch.cat([torch.unsqueeze(depth_tgt_cam_points[iii,p2[iii,:,0],p2[iii,:,1]],dim=0) for iii in range(depth_tgt_cam_points.shape[0])], dim=0)
        xyz_corresp1 = torch.cat([torch.unsqueeze(depth_ref_cam_points[iii,corresp_p1[iii,:,0],corresp_p1[iii,:,1]],dim=0) for iii in range(depth_ref_cam_points.shape[0])], dim=0)
        xyz_corresp2 = torch.cat([torch.unsqueeze(depth_ref_cam_points[iii,corresp_p2[iii,:,0],corresp_p2[iii,:,1]],dim=0) for iii in range(depth_ref_cam_points.shape[0])], dim=0)
        #pdb.set_trace()
        d_p12 = torch.norm((xyz_p2 - xyz_p1), p=2, dim=2).type(torch.DoubleTensor)
        d_corresp12 = torch.norm((xyz_corresp2 - xyz_corresp1), p=2, dim=2).type(torch.DoubleTensor)

        if self.opt.use_dist_normalization and not self.opt.use_rel_dist_normalization:
            d_p12_normalized = d_p12.div(torch.sum(d_p12)) 
            d_corresp12_normalized = d_corresp12.div(torch.sum(d_corresp12)) + 0.000000001

        elif self.opt.use_rel_dist_normalization:
            d_p12_normalized = d_p12.div(d_p12 + d_corresp12 + 0.000000001)
            d_corresp12_normalized = d_corresp12.div(d_p12 + d_corresp12 + 0.000000001)

        else:
            d_p12_normalized = d_p12
            d_corresp12_normalized = d_corresp12

        d_loss = torch.sum(torch.abs(d_p12_normalized - d_corresp12_normalized))

        if self.opt.use_rel_dist_normalization:
            d_loss = d_loss
        elif self.opt.loss_divisor_1:
            d_loss = d_loss * (1/torch.sum(d_p12_normalized+d_corresp12_normalized))# + 1/torch.sum(weights_from_embs_iii))
        elif self.opt.loss_divisor_2:
            d_loss = d_loss / (torch.sum((d_p12_normalized+d_corresp12_normalized)))# * torch.sum(weights_from_embs_iii))
        else:
            d_loss = d_loss /torch.sum((d_p12_normalized+d_corresp12_normalized))


        if self.opt.loss_percentage:
            d_loss = torch.sum((torch.abs(d_p12_normalized - d_corresp12_normalized)).div(torch.abs(d_p12_normalized + d_corresp12_normalized + eps)))
        
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](inputs["color", 0, 0])
        outputs = self.models["depth"](features)

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, d_loss)
        losses['iso_loss'] = iso_loss
        return outputs, losses


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, d_loss):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        loss = 0
        source_scale = 0

        disp = outputs[("disp", 0)]
        color = inputs[("color", 0, 0)]
        target = inputs[("color",1, 0)]

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += d_loss
        #loss += self.opt.disparity_smoothness * smooth_loss
        total_loss += loss
        losses["loss/{}".format(0)] = loss

        losses["loss"] = total_loss
        losses["smooth_loss"] = self.opt.disparity_smoothness * smooth_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())


    def plot_images(self, img_list, emb_list):
        def init():
            img.set_data((np.concatenate((np.clip(1-img_list[0],a_min=0, a_max=1), emb_list[0]),axis=0)*255).astype(np.uint32))
            return (img,)

        def animate(i):
            img.set_data((np.concatenate((np.clip(1-img_list[i],a_min=0, a_max=1), emb_list[i]),axis=0)*255).astype(np.uint32))
            return (img,)

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = ax.imshow((np.concatenate((np.clip(1-img_list[0],a_min=0, a_max=1), emb_list[0]),axis=0)*255).astype(np.uint32));
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(img_list), interval=60, blit=True)
        return anim

    def normalize(self, x):
        return (x-x.min())/(x.max()-x.min())

    def plot_only_depth(self, img_list):
        def init():
            img.set_data(img_list[0])
            return (img,)

        def animate(i):
            img.set_data(img_list[i])
            return (img,)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(w,h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = ax.imshow(img_list[0]);
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(img_list), interval=60, blit=True)
        return anim
        
    def save_depth_video(self):
        self.set_eval()
        root_anim_dir = self.log_path + '/' #model_name_temp + '/' #self.opt.anim_dir + '/' + self.model_name_temp[-24:] + '/'
        cm = plt.get_cmap('plasma')

        loaders = {'vid_val':self.vid_loader_val,'vid_test':self.vid_loader_test}
        
        for lkey in loaders.keys():
            temp_loader = loaders[lkey]
            
            input_images = np.zeros((min(50,len(temp_loader)),self.opt.height,self.opt.width,3))
            seq_depths_plasma = np.zeros((min(50,len(temp_loader)),self.opt.height,self.opt.width,3)) #np.zeros((len(self.train_loader_vid),384,1024,3))
        
            for batch_idx, inputs in enumerate(temp_loader):
                with torch.no_grad():
                    if batch_idx<min(50,len(temp_loader)):
                        
                        for key, ipt in inputs.items():
                            inputs[key] = ipt.to(self.device)

                        tgt_features = self.models["encoder"](inputs["color", 0, 0])
                        _,depth_tgt_list = disp_to_depth(self.models["depth"](tgt_features)['disp', 0], self.opt.min_depth, self.opt.max_depth)

                        #pdb.set_trace()
                        #seq_depths[batch_idx,:,:,0] = seq_depths[batch_idx,:,:,1] = seq_depths[batch_idx,:,:,2] = np.squeeze(normalize_depth(outputs[("depth", 0, 0)][0]).data.cpu().numpy())
                        
                        seq_depths_plasma[batch_idx,:,:,:] = cm(1-(np.tanh(3*np.squeeze(depth_tgt_list.data.cpu().numpy()))))[:,:,0:3]
                        #seq_depths_plasma[batch_idx,:,:,:] = cm(1-(np.squeeze(normalize_depth(depth_tgt_list).data.cpu().numpy())))[:,:,0:3]
                        
                        input_images[batch_idx,:,:,:] = (1-np.squeeze(inputs['color',0,0].permute(0,2,3,1).data.cpu().numpy()))
                    else:
                        break
            
            img_list = list(self.normalize(seq_depths_plasma))
            input_images_nm = list(self.normalize(input_images))
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=12,  bitrate=1800)
            self.plot_images(input_images_nm,img_list).save(root_anim_dir+'/anim_embeddings_'+ str(self.opt.model_name) + '_' + str(lkey) + '_' + str(self.epoch).zfill(3)+'.mp4', writer=writer)
        self.set_train() 


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        cm = plt.get_cmap('plasma')
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        #pdb.set_trace()
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                frame_id = 0
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)

                #writer.add_image("disp_{}/{}".format(s, j), cm(1-(np.squeeze(normalize_depth(outputs[("disp", s)][j]).data.cpu().numpy())))[:,:,0:3], self.step)
                writer.add_image("disp_{}/{}".format(s, j), torch.from_numpy(cm((np.squeeze(normalize_depth(outputs[("disp", s)][j]).data.cpu().numpy())))[:,:,0:3]).permute(2,0,1), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
