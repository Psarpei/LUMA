import numpy as np
import torch
from . import networks
from .pfm import ParametricFaceModel
from util.nvdiffrast import MeshRenderer

class FaceReconModel():
    def __init__(self, fr_ckpt_path, pfm_ckpt_path, device):
        """Initialize this model class.

        Parameters:
            fr_ckpt_path -- checkpoint path
            pfm_ckpt_path -- parametric face model path
            device -- device to use 'cuda' or 'cpu'
        """
        
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']
        self.device = device

        self.net_recon = networks.define_net_recon(
            net_recon='resnet50', use_last_fc=False, init_path=None
        )

        state_dict = torch.load(fr_ckpt_path, map_location=self.device)
        self.net_recon.load_state_dict(state_dict['net_recon'])

        self.facemodel = ParametricFaceModel(
            ckpt_path=pfm_ckpt_path, camera_distance=10.0, focal=1015.0, center=112.0,
            is_train=False
        )
        self.facemodel.to(self.device)
        
        fov = 2 * np.arctan(112.0 / 1015.0) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=5.0, zfar=15.0, rasterize_size=int(2 * 112), use_opengl=False
        )

        self.parallelize()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def proj_img_to_3d(self, img_tensor, use_exp=True):
        output_coeff = self.net_recon(img_tensor)
        pred_face_shape, pred_pose, pred_gamma_coef, pred_tex_coef = \
            self.facemodel.compute_shape_pose(output_coeff, use_exp=use_exp)

        return pred_face_shape, pred_pose, pred_gamma_coef, pred_tex_coef

    def proj_3d_to_img(self, face_shape, pose, gamma_coef, tex_coef=None):
        pred_vertex, pred_color, pred_lm = \
            self.facemodel.compute_for_render_from_shape(face_shape, pose, gamma_coef, tex_coef)
        pred_mask, _, pred_face = self.renderer(
            pred_vertex, self.facemodel.face_buf, feat=pred_color)

        return pred_face.detach().cpu(), pred_mask.detach().cpu(), pred_lm.detach().cpu()

    def forward(self):
        output_coeff = self.net_recon(self.input_img)
        pred_face_shape, pred_vertex, pred_tex, pred_color, pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
        pred_mask, _, pred_face = self.renderer(
            pred_vertex, self.facemodel.face_buf, feat=pred_color)
        
        return output_coeff.detach().cpu(), pred_face_shape.detach().cpu(), pred_face.detach().cpu(), pred_mask.detach().cpu(), pred_lm.detach().cpu()

    def parallelize(self):
        for name in self.parallel_names:
            if isinstance(name, str):
                module = getattr(self, name)
                setattr(self, name, module.to(self.device))

    def eval(self):
        """Make models eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()
