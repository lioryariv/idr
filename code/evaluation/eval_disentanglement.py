import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

import utils.general as utils
import utils.plots as plt
from utils import rend_util

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    timestamp = '2020'
    checkpoint = '2000'

    expname = conf.get_string('train.expname')

    geometry_id = kwargs['geometry_id']
    appearance_id = kwargs['appearance_id']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir_geometry = os.path.join('../', exps_folder_name, expname + '_{0}'.format(geometry_id))
    expdir_appearance = os.path.join('../', exps_folder_name, expname + '_{0}'.format(appearance_id))
    evaldir = os.path.join('../', evals_folder_name, expname + '_{0}_{1}'.format(geometry_id, appearance_id))
    utils.mkdir_ifnotexists(evaldir)

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    # Load geometry network model
    old_checkpnts_dir = os.path.join(expdir_geometry, timestamp, 'checkpoints')
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', checkpoint + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])

    # Load rendering network model
    model_fake = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model_fake.cuda()
    old_checkpnts_dir = os.path.join(expdir_appearance, timestamp, 'checkpoints')
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', checkpoint + ".pth"))
    model_fake.load_state_dict(saved_model_state["model_state_dict"])

    model.rendering_network = model_fake.rendering_network

    dataset_conf = conf.get_config('dataset')
    dataset_conf['scan_id'] = geometry_id
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(False, **dataset_conf)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  collate_fn=eval_dataset.collate_fn
                                                  )
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res

    ####################################################################################################################
    print("evaluating...")

    model.eval()

    gt_pose = eval_dataset.get_gt_pose(scaled=True).cuda()
    gt_quat = rend_util.rot_to_quat(gt_pose[:, :3, :3])
    gt_pose_vec = torch.cat([gt_quat, gt_pose[:, :3, 3]], 1)

    indices_all = [11, 16, 34, 28, 11]
    pose = gt_pose_vec[indices_all, :]
    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)

    n_inter = 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * t_in[-1]).astype(np.float32)

    scales = np.array([4.2, 4.2, 3.8, 3.8, 4.2]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type='periodic')
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose[:, :4].detach().cpu().numpy(), bc_type='periodic')
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    q_new = torch.from_numpy(q_new).cuda().float()

    images_dir = '{0}/novel_views_rendering'.format(evaldir)
    utils.mkdir_ifnotexists(images_dir)

    indices, model_input, ground_truth = next(iter(eval_dataloader))

    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        torch.cuda.empty_cache()

        new_q = new_q.unsqueeze(0)
        new_t = -rend_util.quat_to_rot(new_q)[:, :, 2] * scale

        new_p = torch.eye(4).float().cuda().unsqueeze(0)
        new_p[:, :3, :3] = rend_util.quat_to_rot(new_q)
        new_p[:, :3, 3] = new_t

        sample = {
            "object_mask": torch.zeros_like(model_input['object_mask']).cuda().bool(),
            "uv": model_input['uv'].cuda(),
            "intrinsics": model_input['intrinsics'].cuda(),
            "pose": new_p
        }

        split = utils.split_input(sample, total_pixels)
        res = []
        for s in split:
            out = model(s)
            res.append({
                'rgb_values': out['rgb_values'].detach(),
            })

        batch_size = 1
        model_outputs = utils.merge_output(res, total_pixels, batch_size)
        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

        rgb_eval = (rgb_eval + 1.) / 2.
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
        img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % i))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--geometry_id', type=int, default=65, help='The scan id of the learned geometry.')
    parser.add_argument('--appearance_id', type=int, default=110, help='The scan id of the learned appearance.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             exps_folder_name='trained_models',
             evals_folder_name='evals_disentanglement',
             geometry_id=opt.geometry_id,
             appearance_id=opt.appearance_id
             )
