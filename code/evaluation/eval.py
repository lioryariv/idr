import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_cameras = kwargs['eval_cameras']
    eval_rendering = kwargs['eval_rendering']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(eval_cameras, **dataset_conf)

    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()
    if eval_cameras:
        num_images = len(eval_dataset)
        pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
        pose_vecs.weight.data.copy_(eval_dataset.get_pose_init())

        gt_pose = eval_dataset.get_gt_pose()

    if eval_rendering:
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']

    if eval_cameras:
        data = torch.load(os.path.join(old_checkpnts_dir, 'CamParameters', str(kwargs['checkpoint']) + ".pth"))
        pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

    ####################################################################################################################
    print("evaluating...")

    model.eval()
    if eval_cameras:
        pose_vecs.eval()

    with torch.no_grad():
        if eval_cameras:
            gt_Rs = gt_pose[:, :3, :3].double()
            gt_ts = gt_pose[:, :3, 3].double()

            pred_Rs = rend_util.quat_to_rot(pose_vecs.weight.data[:, :4]).cpu().double()
            pred_ts = pose_vecs.weight.data[:, 4:].cpu().double()

            R_opt, t_opt, c_opt, R_fixed, t_fixed = get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts)

            cams_transformation = np.eye(4, dtype=np.double)
            cams_transformation[:3,:3] = c_opt * R_opt
            cams_transformation[:3,3] = t_opt

        mesh = plt.get_surface_high_res_mesh(
            sdf=lambda x: model.implicit_network(x)[:, 0],
            resolution=kwargs['resolution']
        )

        # Transform to world coordinates
        if eval_cameras:
            mesh.apply_transform(cams_transformation)
        else:
            mesh.apply_transform(scale_mat)

        # Taking the biggest connected component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_clean = components[areas.argmax()]
        mesh_clean.export('{0}/surface_world_coordinates_{1}.ply'.format(evaldir, epoch), 'ply')

    if eval_rendering:
        images_dir = '{0}/rendering'.format(evaldir)
        utils.mkdir_ifnotexists(images_dir)

        psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()

            if eval_cameras:
                pose_input = pose_vecs(indices.cuda())
                model_input['pose'] = pose_input
            else:
                model_input['pose'] = model_input['pose'].cuda()

            split = utils.split_input(model_input, total_pixels)
            res = []
            for s in split:
                out = model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

            rgb_eval = (rgb_eval + 1.) / 2.
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % indices[0]))

            rgb_gt = ground_truth['rgb']
            rgb_gt = (rgb_gt + 1.) / 2.
            rgb_gt = plt.lin2img(rgb_gt, img_res).numpy()[0]
            rgb_gt = rgb_gt.transpose(1, 2, 0)

            mask = model_input['object_mask']
            mask = plt.lin2img(mask.unsqueeze(-1), img_res).cpu().numpy()[0]
            mask = mask.transpose(1, 2, 0)

            rgb_eval_masked = rgb_eval * mask
            rgb_gt_masked = rgb_gt * mask

            psnr = calculate_psnr(rgb_eval_masked, rgb_gt_masked, mask)
            psnrs.append(psnr)

        psnrs = np.array(psnrs).astype(np.float64)
        print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))


def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts,):
    ''' Align predicted pose to gt pose and print cameras accuracy'''

    # find rotation
    d = pred_Rs.shape[-1]
    n = pred_Rs.shape[0]

    Q = torch.addbmm(torch.zeros(d, d, dtype=torch.double), gt_Rs, pred_Rs.transpose(1, 2))
    Uq, _, Vq = torch.svd(Q)
    sv = torch.ones(d, dtype=torch.double)
    sv[-1] = torch.det(Uq @ Vq.transpose(0, 1))
    R_opt = Uq @ torch.diag(sv) @ Vq.transpose(0, 1)
    R_fixed = torch.bmm(R_opt.repeat(n, 1, 1), pred_Rs)

    # find translation
    pred_ts = pred_ts @ R_opt.transpose(0, 1)
    c_opt = cp.Variable()
    t_opt = cp.Variable((1, d))

    constraints = []
    obj = cp.Minimize(cp.sum(
        cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) @ t_opt), axis=1)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    t_fixed = c_opt.value * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) * t_opt.value

    # Calculate transaltion error
    t_error = np.linalg.norm(t_fixed - gt_ts.numpy(), axis=-1)
    t_error = t_error
    t_error_mean = np.mean(t_error)
    t_error_medi = np.median(t_error)

    # Calculate rotation error
    R_error = compare_rotations(R_fixed, gt_Rs)

    R_error = R_error.numpy()
    R_error_mean = np.mean(R_error)
    R_error_medi = np.median(R_error)

    print('CAMERAS EVALUATION: R error mean = {0} ; t error mean = {1} ; R error median = {2} ; t error median = {3}'
          .format("%.2f" % R_error_mean, "%.2f" % t_error_mean, "%.2f" % R_error_medi, "%.2f" % t_error_medi))

    # return alignment and aligned pose
    return R_opt.numpy(), t_opt.value, c_opt.value, R_fixed.numpy(), t_fixed

def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')
    parser.add_argument('--eval_cameras', default=False, action="store_true", help='If set, evaluate camera accuracy of trained cameras.')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_cameras=opt.eval_cameras,
             eval_rendering=opt.eval_rendering
             )
