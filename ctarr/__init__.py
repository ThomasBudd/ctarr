import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from ctarr.model.network import UNet
from ctarr.model import parameters
from ctarr.utils.io import dataset, save_nii, save_pkl
from ctarr.utils.torch_affine_transf import AffineTransf
from skimage.measure import label
import os


class CTARR():
    
    def __init__(self,
                 bounding_box_name=None,
                 allow_z_flipping=True,
                 allow_x_flipping=False,
                 allow_y_flipping=False,
                 allow_xy_rotation=True,
                 additional_margin=10,
                 path_to_file_storage=None):
        self.allow_z_flipping = allow_z_flipping
        self.allow_x_flipping = allow_x_flipping
        self.allow_y_flipping = allow_y_flipping
        self.allow_xy_rotation = allow_xy_rotation
        self.path_to_file_storage = path_to_file_storage or os.path.join(__path__[0], 'file_storage') 
        self.additional_margin = additional_margin
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'   
        
        print(f'Using the following path as file storage: {self.path_to_file_storage}')
        print(f'Using the following device for network inference: {self.device}')
        
        self._initialise_network()
        
        if bounding_box_name is not None:
            self.load_bounding_box(bounding_box_name)
        else:
            print('No bounding box selected so far. Here is a list you can pick from.')
            self.print_available_bounding_boxes()
            print('Use load_bouding_box(NAME) for loading.')
        
        # the spacing, we will need it often
        self.sp = np.array(parameters.TARGET_SPACING)
        
        # load the atlas
        self.seg_atlas = np.load(os.path.join(self.path_to_file_storage, 'seg_atlas.npy'))
        self.seg_atlas = self.seg_atlas.astype(np.float16) / 255
        self.seg_atlas = torch.from_numpy(self.seg_atlas).to(self.device)
        # compute the correponding landmarks
        self.lms_atlas = self._extract_lms(self.seg_atlas)
        
        self.atlas_cs = self._get_cs(self.seg_atlas)
        self.n_classes = self.seg_atlas.shape[0]
        self.atlas_clp = np.stack([np.zeros(3), np.array(self.seg_atlas.shape)[1:]]).astype(int)
    
    def _get_cs(self, arr):
        shp = np.array(arr.shape[-3:])
        return np.stack([-0.5 * shp * self.sp, 0.5 * shp * self.sp])
    
    def _interp(self, arr, size=None, scale_factor=None, mode='nearest'):
        
        is_np = isinstance(arr, np.ndarray)
        ndim_in = len(arr.shape)
        assert ndim_in in [3,4], f'Input must be either 3d or 4d, but got {ndim_in}'
        
        if is_np:
            arr = torch.from_numpy(arr).to(self.device)
        
        while arr.ndim < 5:
            arr = arr[None]
            
        if size is not None:
            size = tuple([int(s) for s in size])
        
        if scale_factor is not None:
            scale_factor = tuple([float(s) for s in scale_factor])
        
        arr = F.interpolate(arr,
                            size=size,
                            scale_factor=scale_factor,
                            mode=mode)
        
        while arr.ndim > ndim_in:
            arr = arr[0]
        
        if is_np:
            arr = arr.cpu().numpy()
        
        return arr
        
    def _initialise_network(self):        
        # create torch module
        self.network = UNet()
        # load network weights
        self.network.load_state_dict(torch.load(os.path.join(self.path_to_file_storage, 'network_weights')))
        # to device
        self.network = self.network.to(self.device)
        
        # disable gradients
        for p in self.network.parameters():
            p.requires_grad = False
    
    def _to_4d(self, arr):
        
        if arr.ndim == 3:
            arr = arr[None]
        elif arr.ndim != 4:
            raise ValueError(f"input array must be 3d or 4d. Got shape {arr.shape}")
        
        return arr
    
    def _preprocess_image(self, scan):
        
        # get image and spacing
        im = self._to_4d(scan['image'])
        im = torch.from_numpy(np.copy(im)).float().to(self.device)
        spacing = np.array(scan['spacing'])
        
        # resize to target spacing
        scale_factor = (spacing / self.sp).tolist()
        # im = F.interpolate(im.unsqueeze(0),
        #                    scale_factor=scale_factor,
        #                    mode='trilinear')[0]
        im = self._interp(im, scale_factor=scale_factor, mode='trilinear')
        
        # apply windowing and scaling
        im = torch.cat([(im.clip(*win) - scale[0])/scale[1] 
                        for win, scale in zip(parameters.CT_WINDOWS,
                                              parameters.CT_SCALINGS)])
        
        # return as torch tensor
        return im
    
    def _evaluate_network(self, prep_im):
        # evaluates the preprocessed image by applying sliding window
        
        # get some parameters for sliding window
        shp_in = np.array(prep_im.shape[1:])
        ps = np.array(parameters.PATCH_SIZE).astype(int)
        ovlp = parameters.PATCH_OVERLAP
        pw = torch.from_numpy(parameters.PATCH_WEIGHT).to(self.device)
        
        # pad the image in case it is smaller than the patch size
        pad = [0, ps[2] - shp_in[2],
               0, ps[1] - shp_in[1],
               0, ps[0] - shp_in[0]]
        pad = np.maximum(pad, 0).tolist()
        prep_im = F.pad(prep_im, pad).type(torch.float)
        shp = np.array(prep_im.shape[1:])
        
        # create the list of zxy coords for cropping of patches
        n_ptchs = np.ceil((shp - ps) / (ovlp * ps)).astype(int) + 1        
        grids = [np.linspace(0, shp[i] - ps[i], n_ptchs[i]).astype(int) for i in range(3)]        
        coords_list = []
        for z in grids[0]:
            for x in grids[1]:
                for y in grids[2]:
                    coords_list.append((z,x,y))
            
        # reserve storage for results
        pred = torch.zeros((self.n_classes+1, *shp),
                           device='cuda',
                           dtype=torch.float)
        ovlp = torch.zeros((1, *shp),
                           device='cuda',
                           dtype=torch.float)
        with torch.cuda.amp.autocast():
            for z,x,y in coords_list:            
                # crop image
                crop = prep_im[:, z:z+ps[0], x:x+ps[1], y:y+ps[2]]
                # eval network
                out = self.network(crop.unsqueeze(0))[0]
                # fill with weights in storages
                pred[:, z:z+ps[0], x:x+ps[1], y:y+ps[2]] += out * pw
                ovlp[:, z:z+ps[0], x:x+ps[1], y:y+ps[2]] += pw
            
        # if the image was smaller than the patch size we have to crop again
        pred = pred[:, :shp_in[0], :shp_in[1], :shp_in[2]].float()
        ovlp = ovlp[:, :shp_in[0], :shp_in[1], :shp_in[2]].float()
        
        # correction with overlap and application of softmax
        soft_seg = (pred / ovlp).softmax(0)
        
        # we only need the foreground classes
        return soft_seg[1:]
    
    def _extract_lms(self, soft_seg):#, orig_shape, orig_spaing):
        
        # get shape and length per axis in real world units
        shp = soft_seg.shape[1:]
        len_per_axis = np.array(shp) * self.sp
        
        # create corresponding coordinate grid
        z, x, y = [torch.linspace(-l/2, l/2, s).cuda() for s, l in zip(shp, len_per_axis)]
        
        # mass of each ROI in number of voxels
        masses = soft_seg.float().flatten(1).sum(1, keepdims=True).clip(1e-5)
        
        # marginal distributions
        # this is simpler than creating the meshgrid
        pz = soft_seg.sum((2,3)) / masses
        px = soft_seg.sum((1,3)) / masses
        py = soft_seg.sum((1,2)) / masses
        
        # center of masses computed via axes means
        z_mean = (pz * z).sum(1)        
        x_mean = (px * x).sum(1)        
        y_mean = (py * y).sum(1)        
        coms = torch.stack([z_mean, x_mean, y_mean], -1)
        
        # correct masses with volume per voxel
        masses *= np.prod(self.sp) 
        
        # concat masses and return as numpy
        lms = torch.cat((coms, masses), 1)
        return lms.cpu().numpy()
    
    def _rot_landmarks(self, lms, k_rot:int):

        k = k_rot % 4
        
        Z, X, Y = lms[:, 0], lms[:, 1], lms[:, 2]
        
        if k == 0:            
            return lms
        
        elif k == 1:
            return np.stack([Z, -1*Y, X], -1)
        
        elif k == 2:
            return np.stack([Z, -1*X, -1*Y], -1)
        
        elif k ==3:
            return np.stack([Z, Y, -1*X], -1)    
    
    def _rot_arr(self, arr, k_rot):
        
        if isinstance(arr, np.ndarray):
            return np.rot90(arr, k_rot, (-2, -1))
        elif torch.is_tensor(arr):
            return torch.rot90(arr, k_rot, (-2, -1))
        else:
            raise TypeError(f'Expected np.ndarray or torch.tensor, but got {type(arr)}')
        
    def _flip_arr(self, arr, flp):
        
        dim = tuple([i+1  for i, f in enumerate(flp) if f])
        
        if len(dim) > 0:        
            if isinstance(arr, np.ndarray):
                return np.flip(arr, dim)
            elif torch.is_tensor(arr):
                return torch.flip(arr, dim)
            else:
                raise TypeError(f'Expected np.ndarray or torch.tensor, but got {type(arr)}')
        else:
            return arr

    def _get_scan_atlas_crops(self, scan_cs_transl):
        atlas_inds = np.round((scan_cs_transl - self.atlas_cs[0]) / self.sp).astype(int)
        atlas_inds_clp = np.stack([atlas_inds[0].clip(*self.atlas_clp),
                                   atlas_inds[1].clip(*self.atlas_clp)])
        atlas_crop = self.seg_atlas[:, atlas_inds_clp[0, 0]:atlas_inds_clp[1, 0], 
                                    atlas_inds_clp[0, 1]:atlas_inds_clp[1, 1], 
                                    atlas_inds_clp[0, 2]:atlas_inds_clp[1, 2]]

        pad = (atlas_inds_clp[0, 2] - atlas_inds[0, 2],
               atlas_inds[1, 2] - atlas_inds_clp[1, 2],
               atlas_inds_clp[0, 1] - atlas_inds[0, 1],
               atlas_inds[1, 1] - atlas_inds_clp[1, 1],
               atlas_inds_clp[0, 0] - atlas_inds[0, 0],
               atlas_inds[1, 0] - atlas_inds_clp[1, 0])
        atlas_crop_pad = F.pad(atlas_crop, pad)  
        return atlas_crop_pad
        
    def _loss_fctn(self, seg1, seg2):
        ovlp = (seg1 * seg2).sum((2,3,4))
        masses = (seg1 + seg2).sum((2,3,4)).clip(1e-5)
        dsc = torch.mean(2 * ovlp / masses)
        return 1 - dsc    
    
    def _iterativ_matching(self,
                           seg_mv, 
                           b_init,
                           lr0=parameters.LR0, 
                           lr_fac=parameters.LR_FAC, 
                           n_lr_red=parameters.N_LR_RED,
                           del_loss=parameters.DEL_LOSS):
        
        # crop and add batch axis
        scan_cs_transl = self._get_cs(seg_mv) + b_init
        yb = self._get_scan_atlas_crops(scan_cs_transl)[None]
        xb = seg_mv[None]
        
        len_per_ax = np.array(xb.shape[2:]) * self.sp
        # get object to perform the affine transf and SGD optimizer
        affine_transf = AffineTransf(len_per_ax=len_per_ax)        
        affine_transf = affine_transf.to(self.device)
        opt = torch.optim.SGD(affine_transf.parameters(), 
                              lr=lr0, 
                              momentum=0, 
                              weight_decay=0)
        
        
        # iteration variables
        stop, prev_loss, fac_counter = False, None, 0
        with torch.cuda.amp.autocast():
            while not stop:    
                affine_transf.zero_grad()
                out = affine_transf(xb)
                loss_val = self._loss_fctn(out, yb)
                loss_val.backward()
                current_loss = loss_val.detach().item()
                if prev_loss is not None:
                    if prev_loss - current_loss < del_loss:
                        opt.param_groups[0]['lr'] *= lr_fac
                        fac_counter += 1
                        if fac_counter == n_lr_red:
                            stop = True
                opt.step()
                prev_loss = current_loss

        # return affine parameters
        return affine_transf.get_lin_params()
    
    def _compute_transf(self, soft_seg):
        
        # landmarks of the scan
        lms_scan = self._extract_lms(soft_seg)
        
        # landmarks of the atlas
        y = self.lms_atlas[:, :3]
        x = lms_scan[:, :3]
        
        # weight per landmark
        w = lms_scan[:, 3:] / self.lms_atlas[:, 3:]
        # inference of orientation is only performed if enough segmentations
        # are sufficiently contained
        infere_orientation = np.sum(w > 0.25) > 3
        # normalize to sum one
        w /= w.sum(0, keepdims=True)        
        
        if infere_orientation:
        
            transf_list = []
            wmse_list = []
            
            for k_rot in range(4):
                # try all for rotations in xy plane            
                x_rot = self._rot_landmarks(x, k_rot)
                x_bar, y_bar = np.sum(x_rot * w, 0), np.sum(y * w, 0)  
                x_var = np.sum(w * (x_rot - x_bar) ** 2, 0)
                xy_covar = np.sum(w * (x_rot - x_bar) * (y - y_bar), 0)
                a = xy_covar / x_var
                a = np.sign(a) if self.allow_z_flipping else 1
                for i, b in enumerate((self.allow_z_flipping, 
                                       self.allow_x_flipping, 
                                       self.allow_y_flipping)):
                    if not b:
                        a[i] = 1
                b = y_bar - a * x_bar
                
                # compute also the match between the landmark pairs as weighted MSE
                x_hat = a * x_rot + b
                wmse =  np.sum(w * (y - x_hat)**2)
                flp = tuple(a < 0)
                transf_list.append((a<0, b))
                wmse_list.append(wmse)
            
            k_opt = np.argmin(wmse_list)
            flp, b_init = transf_list[k_opt]
        
        else:            
            x_bar, y_bar = np.sum(x * w, 0), np.sum(y * w, 0)              
            k_opt, flp, b_init = 0, (False, False, False), y_bar - x_bar
        
        # create input for 
        scale, transl = self._iterativ_matching(self._flip_arr(self._rot_arr(soft_seg,
                                                                             k_opt),
                                                               flp),
                                                b_init)
        
        return (k_opt, flp, scale, scale * b_init + transl)

    def _get_transf_from_scan(self, scan):
        
        if 'transf' not in scan:
            # do all steps to get the affine transformation
            prep_im = self._preprocess_image(scan)
            soft_seg = self._evaluate_network(prep_im)
            transf = self._compute_transf(soft_seg)
            scan['transf'] = transf
            
        return scan['transf']
    
    def crop(self,
             arr,
             scan,
             padding_val=None,
             target_spacing=None,
             z_only=False,
             mode='nearest'):
        
        arr = self._to_4d(arr)
        
        transf = self._get_transf_from_scan(scan)
        
        # rotate
        arr = self._rot_arr(arr, transf[0])
        # apply flipping
        arr = self._flip_arr(arr, transf[1])
        
        # translate bounding boxes
        transf_bb = (self.bounding_boxes - transf[3]) / transf[2]
        
        # to voxels indices        
        sp, shp = np.array(scan['spacing']), np.array(arr.shape[1:])
        transf_bb = np.round(transf_bb / sp + shp/2)
        
        if z_only:
            # maybe crop only in z direction
            transf_bb[:, 0, 1:] = 0
            transf_bb[:, 1, 1:] = shp[1:]
        
        # ensure we don't access indices outside the array
        transf_bb_clp = np.minimum(np.maximum(transf_bb, 0), shp).astype(int)
        
        # store for later
        scan['shape_before_cropping'] = np.array(arr.shape)[1:]
        scan['cropping_coordinates'] = transf_bb_clp
        
        arr_crops = [arr[:, l[0]:u[0], l[1]:u[1], l[2]:u[2]] for l, u in transf_bb_clp]
        
        apply_padding = padding_val is not None
        apply_resizing = target_spacing is not None
        
        if apply_padding:
            if padding_val == 'min':
                padding_val = np.min(arr, (1,2,3))
            
            # lower and upper padding for each box and axis
            pad_l = (transf_bb_clp[:, 0] - transf_bb[:, 0]).astype(int)
            pad_u = (transf_bb[:, 1] - transf_bb_clp[:, 1]).astype(int)
            
            scan['pad_l'] = pad_l
            scan['pad_u'] = pad_u
            
            arr_crops_padded = []
            
            if isinstance(arr, np.ndarray):
                for arr_crop, pl, pu in zip(arr_crops, pad_l, pad_u):
                    pad = tuple([(0, 0)] + [(l, u) for l, u in zip(pl, pu)])
                    arr_crop_pad = np.pad(arr_crop,
                                          pad,
                                          constant_values=padding_val)
                    arr_crops_padded.append(arr_crop_pad)
            elif torch.is_tensor(arr):
                for arr_crop, pl, pu in zip(arr_crops, pad_l, pad_u):
                    pad = (pl[2], pu[2], pl[1], pu[1], pl[0], pu[0])
                    if np.isscalar(padding_val):
                        arr_crop_pad = F.pad(arr_crop,
                                             pad,
                                             'constant',
                                             padding_val)
                    else:
                        arr_crop_pad = torch.stack([F.pad(arr_crop[i],
                                                       pad,
                                                       'constant',
                                                       padding_val[i])
                                                    for i in range(len(arr_crop))])
                        
                    arr_crops_padded.append(arr_crop_pad)
            else:
                raise TypeError(f"Expected input of type np.ndarray or torch.tensor, but got {type(arr)}")
                
            arr_crops = arr_crops_padded
        
        if apply_resizing:
            arr_crops_resized = []
            scan['shapes_before_resizing'] = np.stack([a.shape[-3:] for a in arr_crops]).tolist()
            
            tar_sp = np.array(target_spacing)
            if apply_padding and not z_only:
                # here we compute the original size 
                lens_per_ax = self.bounding_boxes[:, 1] - self.bounding_boxes[:, 0]
                sizes = np.round(lens_per_ax / tar_sp).astype(int)
                
                for arr_crop, size in zip(arr_crops, sizes):
                    # if is_np:
                        # arr_crop = torch.from_numpy(arr_crop).to(self.device)
                    
                    # arr_crop = F.interpolate(arr_crop[None],
                                             # size=tuple(size))[0]
                    
                    arr_crop = self._interp(arr_crop, size=tuple(size), mode=mode)
                    # if is_np:
                        # arr_crop = arr_crop.cpu().numpy()
                    arr_crops_resized.append(arr_crop)
            else:
                scale_factor = tuple(sp / tar_sp)
                for arr_crop in arr_crops:
                    # if is_np:
                        # arr_crop = torch.from_numpy(arr_crop).to(self.device)
                    
                    # arr_crop = F.interpolate(arr_crop[None],
                                             # scale_factor=scale_factor)[0]
                    
                    arr_crop = self._interp(arr_crop,
                                            scale_factor=scale_factor,
                                            mode=mode)
                    
                    # if is_np:
                        # arr_crop = arr_crop.cpu().numpy()
                    arr_crops_resized.append(arr_crop)
            
            arr_crops = arr_crops_resized

        return arr_crops        
    
    def __call__(self, scan, padding_val=None, target_spacing=None, z_only=False):   
        
        if 'image' in scan:
            scan['image_crops'] = self.crop(scan['image'], scan, padding_val, target_spacing, z_only=z_only)
        
        if 'label' in scan:
            scan['label_crops'] = self.crop(scan['label'], scan, padding_val, target_spacing, z_only=z_only)
    
    def undo_cropping(self,
                      arr_crops,
                      scan,
                      fill_val='min',
                      mode='nearest'):
        
        # check input        
        if isinstance(arr_crops, np.ndarray):
            arr_crops = [arr_crops]
        
        assert isinstance(arr_crops, (list, tuple)), 'input array must be numpy ndarray, list or tuple'        
        
        n_boxes = len(scan['cropping_coordinates'])
        assert len(arr_crops) == n_boxes, f'Got {n_boxes} bounding boxes, but {len(arr_crops)} crops'
        
        # ensure 4d shape
        arr_crops = [self._to_4d(arr_crop) for arr_crop in arr_crops]
        
        # get filling value for background
        if fill_val == 'min':
            fill_val = min([np.min(crop) for crop in arr_crops])
        
        assert np.isscalar(fill_val), 'fill_val must be \'min\' or scalar value'
        
        # reserve storage and get info
        shp = scan['shape_before_cropping']            
        arr = np.ones((len(arr_crops[0]), *shp)) * fill_val
        
        for i, (crop, (l, u)) in enumerate(zip(arr_crops, scan['cropping_coordinates'])):
            # now crop from the image
            if 'shapes_before_resizing' in scan:
                # in case the crops were resized we invert it again
                size = scan['shapes_before_resizing'][i]
                crop = self._interp(crop, size=size, mode=mode)
            if 'pad_l' in scan:
                # remove the voxels that were created by padding
                l_crp = scan['pad_l'][i]
                u_crp = scan['pad_l'][i] + u - l
                crop = crop[:, l_crp[0]:u_crp[0], l_crp[1]:u_crp[1], l_crp[2]:u_crp[2]]
            
            # fill crop in the full array
            arr[:, l[0]:u[0], l[1]:u[1], l[2]:u[2]] = crop
        
        # flip back
        arr = self._flip_arr(arr, scan['transf'][1])
        
        # rotate back
        arr = self._rot_arr(arr, 4 - scan['transf'][0])
        
        return arr
    
    def _transf_arr(self, 
                    arr, 
                    transf, 
                    current_spacing,
                    target_spacing, 
                    target_coordinate_system,
                    mode='nearest'):        
        
        arr = self._to_4d(arr)
        
        # apply rotation and flipping
        arr = self._rot_arr(arr, transf[0])
        arr = self._flip_arr(arr, transf[1])
        
        # compute coordinate system and translate
        shp = np.array(arr.shape[-3:])
        ccs = np.stack([-1*shp * current_spacing / 2, shp * current_spacing/2], 0)
        ccs_transf = transf[2] * ccs + transf[3]        
        
        # we first compute the position of the array in the target coordinate system
        # resize the image to the matching size and pad it to the right shape
        cl = (ccs_transf[0] - target_coordinate_system[0]) / target_spacing
        cu = cl + (ccs_transf[1] - ccs_transf[0]) / target_spacing
        # coordinates in taget space
        cl, cu = np.round(cl).astype(int), np.round(cu).astype(int)
        # shape for resizing
        new_shp = cu - cl       
        
        # resize array to new shape
        # arr_cuda = torch.from_numpy(np.copy(arr)).cuda().float()
        # arr_cuda = F.interpolate(arr_cuda.unsqueeze(0),
                                                   # size=new_shp.tolist(),
                                                   # mode=mode)[0]
        # arr_new = arr_cuda.cpu().numpy()        
        arr_new = self._interp(np.copy(arr), size=new_shp.tolist(), mode=mode)
        
        # create target array
        target_shp = (target_coordinate_system[1] - target_coordinate_system[0]) / target_spacing
        target_shp = np.round(target_shp).astype(int)
        
        arr_transf = np.zeros((arr.shape[0], *target_shp))
        
        # now we only have to put the array in the right position in arr_transf
        arr_transf[:, cl[0]:cu[0], cl[1]:cu[1], cl[2]:cu[2]] = arr_new
        
        return arr_transf

    def _apply_merging(self, bbl1, bbu1, bbl2, bbu2):
        
        bbl_inters = np.maximum(bbl1, bbl2)
        bbu_inters = np.minimum(bbu1, bbu2)
        
        return np.all(bbl_inters <= bbu_inters)
    
    def _merge_boxes(self, bbl1, bbu1, bbl2, bbu2):
        
        bl = np.minimum(bbl1, bbl2)
        bu = np.maximum(bbu1, bbu2)
        
        return np.stack([bl, bu], 0)
    
    def _update_bounding_boxes(self, bbl, bbu):
        
        for i, bb in enumerate(self.bounding_boxes):
            
            if self._apply_merging(bb[0], bb[1], bbl,  bbu):
                self.bounding_boxes[i] = self._merge_boxes(bb[0], bb[1], bbl,  bbu)
                return
        
        # if we haven't returned by now, we do not merge, but apped instead
        bb_new = np.stack([bbl, bbu])
        if isinstance(self.bounding_boxes, list):
            self.bounding_boxes.append(bb_new)
        else:
            self.bounding_boxes = np.stack([self.bounding_boxes, bb_new])
            

    def infere_bounding_boxes(self, 
                              srcp, 
                              used_classes=None, 
                              margin=0, 
                              threshold=0.01, 
                              allow_multiple_boxes=True):
        
        # this is the dataset we use to infere bounding boxes, but first things first
        ds = dataset(srcp)
        
        # to map a scan to a target coordiante system and compare it with other
        # scans there, we have to agree on a spacing and size of the target coordinate system
        
        # spacing is easy
        target_spacing = ds.get_median_spacing()
        
        # the target volume (in mm) is more difficult. therefore we 
        # compute the transformation and check the coordiantes of the images 
        # in the target coordinate system
        
        transf_list = []
        ccs_transf_list = []
        print('Computing transformations and compute target coordinate system...')
        # iterate over images to compute the transformations (no labels needed)
        for i in tqdm(range(len(ds))):
            scan = ds.__getitem__(i, image_only=True)
            transf = self._get_transf_from_scan(scan)
            transf_list.append(transf)
            
            # check where the coordinate system ends up after transformation.
            # build current cooridante system
            shp = np.array(scan['image'].shape[-3:])
            sp = np.array(scan['spacing'])
            if transf[0] in [1,3]:
                # rotate coordinate system
                shp, sp = shp[[0,2,1]], sp[[0,2,1]]
            
            ccs = np.stack([-1* shp * sp / 2, shp * sp / 2], 0)
            
            # apply transformation
            ccs_transf = transf[2] * ccs + transf[3]
            ccs_transf_list.append(ccs_transf)
            
        # finally the lower and upper coordinate of the target system can be computed
        ccs_transf_list = np.array(ccs_transf_list)
        target_coordinate_system = np.stack([np.min(ccs_transf_list[:, 0], 0),
                                             np.max(ccs_transf_list[:, 1], 0)], 0)
        # length per axis in target coordinate system in mm...
        len_per_ax_tcs = target_coordinate_system[1] - target_coordinate_system[0]
        # ... and voxel
        target_shape = np.round(len_per_ax_tcs / target_spacing).astype(int)
        
        # count how often foreground was seen in each voxel
        ovlps = np.zeros(target_shape)
        counter = 0
        # now the good fun! We iterate over the labels to ensemble them
        print('Transform label to atlas and overlay masks...')
        for i in tqdm(range(len(ds))):
            scan = ds.__getitem__(i, label_only=True)
            arr = self._to_4d(np.round(scan['label']))
            
            if isinstance(used_classes, (list, tuple, np.ndarray)):
                # if we're lazy and want to reduce the number of classes
                # present in the image, we can use such list
                arr_new = np.zeros_like(arr)
                
                for cl in used_classes:
                    arr_new[arr == cl] = 1
                
                arr = arr_new
            else:
                # binarise the label
                arr = (arr > 0).astype(float)
            
            if arr.max() == 0:
                continue
            
            transf = transf_list[i]
            current_spacing = scan['spacing']
            
            lb_atlas = self._transf_arr(arr, 
                                        transf, 
                                        current_spacing,
                                        target_spacing, 
                                        target_coordinate_system)
            # update ovlp array
            ovlps += lb_atlas[0]
            counter += 1
        
        # normalise to [0,1]
        ovlps /= counter
        # this removes some small outliers
        region = (ovlps > threshold).astype(float)
        
        # some ROI might lie appart from each others with no overlap, we will compute seperate
        # bounding boxes in this case
        self.bounding_boxes = []
        if allow_multiple_boxes:
            ccs, nccs = label(region, return_num=True)
        else:
            ccs, nccs = region, 1
        
        for icc in range(1,nccs+1):            
            # compute bounding box for connected component
            Z, X, Y = np.where(ccs == icc)
            bbl = np.array([Z.min(), X.min(), Y.min()])
            bbu = np.array([Z.max()+1, X.max()+1, Y.max()+1])
        
            # recompute to mm spacing
            bbl = bbl * target_spacing + target_coordinate_system[0]
            bbu = bbu * target_spacing + target_coordinate_system[0]
            
            # add margin and map back to coordinate system
            bbl = np.maximum(bbl - margin, target_coordinate_system[0])
            bbu = np.minimum(bbu + margin, target_coordinate_system[1])
            
            self._update_bounding_boxes(bbl, bbu)
        
        self.bounding_boxes = np.array(self.bounding_boxes)
        
        print(f"Inferred a total of {len(self.bounding_boxes)} bounding box(es).")
        
        return ovlps
            
    def store_bounding_box(self, bounding_box_name:str, overwrite=False):
        
        assert hasattr(self, 'bounding_boxes'), 'Can\'t save bounding box, no found/computed'
        assert isinstance(self.bounding_boxes, np.ndarray), 'Bounding boxes must be of type np.ndarray'
        
        path_to_bb = os.path.join(self.path_to_file_storage, f'{bounding_box_name}.npy')
        if os.path.exists(path_to_bb) and not overwrite:
            raise FileExistsError(f'A bounding box file with name {bounding_box_name} already exists. '
                                  'Either choose a different name or pass overwrite=True to this function. '
                                  'Use \'print_available_bounding_boxes\' to check for names.')
        
        np.save(path_to_bb, self.bounding_boxes)
        print(f'stored bounding box(es) at {path_to_bb}')
    
    def load_bounding_box(self, bounding_box_name):
        path_to_bb = os.path.join(self.path_to_file_storage, f'{bounding_box_name}.npy')
        self.bounding_boxes = np.load(path_to_bb)
        self.bounding_boxes[:, 0] -= self.additional_margin
        self.bounding_boxes[:, 1] += self.additional_margin
        print(f'Succesfully loaded bounding box(es) {bounding_box_name}')
    
    def print_available_bounding_boxes(self):
        bb_files = [f.split('.')[0] 
                    for f in os.listdir(self.path_to_file_storage) 
                    if f.endswith('.npy') and not f == 'seg_atlas.npy']
        
        print('Found the following bounding boxes ready to use:')
        print(*bb_files)
    
    def crop_and_store_data(self, srcp, tarp, ext='', min_ax_len=0, 
                            padding_val=None, target_spacing=None):
        
        ds = dataset(srcp)       
        
        n_voxels_old = 0
        n_voxels_new = 0
        
        os.makedirs(os.path.join(tarp, f'images{ext}'), exist_ok=True)
        os.makedirs(os.path.join(tarp, f'fingerprints{ext}'), exist_ok=True)
        if ds.label_folder:
            os.makedirs(os.path.join(tarp, f'labels{ext}'), exist_ok=True)
        
        print(f'Cropping scan from {srcp} and storing them at {tarp} as nifti files')
        
        for scan in tqdm(ds):
            
            sp = np.array(scan['spacing'])
            
            n_voxels_old += int(np.prod(scan['image'].shape[-3:]))
            
            self.__call__(scan, padding_val, target_spacing)
            
            file_name = os.path.basename(scan['path_to_image_file']).split('.')[0]
            
            # get fingerprint, everything that is not array!
            fp = {key: scan[key] for key in scan if key not in ['image', 'label', 'image_crops', 'label_crops']}
            
            save_pkl(fp, os.path.join(tarp, f'fingerprints{ext}', file_name))
            
            for i, arr in enumerate(scan['image_crops']):
                
                if len(scan['image_crops']) == 1:
                    outfile = os.path.join(tarp, f'images{ext}', f'{file_name}.nii.gz')
                else:
                    outfile = os.path.join(tarp, f'images{ext}', f'{file_name}_{i:03d}.nii.gz')
                
                ax_lens = np.array(arr.shape[-3:]) * sp
                if np.min(ax_lens) <= min_ax_len:
                    continue
                
                save_nii(arr, scan['path_to_image_file'], outfile)
                
                n_voxels_new += int(np.prod(arr.shape[-3:]))
            
            if ds.label_folder:
                    
                for i, arr in enumerate(scan['label_crops']):
                    
                    if len(scan['label_crops']) == 1:
                        outfile = os.path.join(tarp, f'labels{ext}', f'{file_name}.nii.gz')
                    else:
                        outfile = os.path.join(tarp, f'labels{ext}', f'{file_name}_{i:03d}.nii.gz')
                    
                    ax_lens = np.array(arr.shape[-3:]) * sp
                    if np.min(ax_lens) <= min_ax_len:
                        continue
                    
                    save_nii(arr, scan['path_to_label_file'], outfile)
        
        print(f'The total number of voxels in the dataset was reduced by a factor of {n_voxels_old/n_voxels_new:.2f}')