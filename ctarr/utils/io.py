import nibabel as nib
import os
import numpy as np
import pickle


def load_pkl(path_to_file):
    if not path_to_file.endswith('.pkl'):
        path_to_file += '.pkl'
    
    with open(path_to_file, 'rb') as file:
        data = pickle.load(file)

    return data

def save_pkl(data, path_to_file):
    if not path_to_file.endswith('.pkl'):
        path_to_file += '.pkl'

    with open(path_to_file, 'wb') as file:
        pickle.dump(data, file)

def _load_nii_file(path_to_nifti):
    
    if not os.path.exists(path_to_nifti):
        raise FileNotFoundError(f'Couldn\'t read {path_to_nifti}, file does not exsist.')
        
    nii_img = nib.load(path_to_nifti)
    sp = nii_img.header['pixdim'][1:4]
    sp = (sp[2], sp[0], sp[1])
    im = nii_img.get_fdata()
    assert im.ndim in [3, 4], "Image from nifti file must be 3d"
    
    if im.ndim == 3:
        im = im[None]
        
    elif im.ndim != 4:
        raise ValueError(f"Image must be 3d or 4d, but got {im.ndim}d.\n"
                         f"Check the file {path_to_nifti}")
    # reorient from typical nifti convention to how we plot the image with
    # matplotlib
    im = np.moveaxis(im, 3, 1)
    im = np.rot90(im, 1, (2,3))[:, ::-1, :, ::-1]
    return im, sp

def load_scan(path_to_image_file=None,
              path_to_label_file=None):
    
    scan = {}
    
    if path_to_image_file is not None:
        im, spacing = _load_nii_file(path_to_image_file)
        
        scan['image'] = im
        scan['spacing'] = spacing
        scan['path_to_image_file'] = path_to_image_file
    
    if path_to_label_file is not None:
    
        lb, spacing = _load_nii_file(path_to_label_file)
        
        if 'spacing' in scan:
            if np.max(np.abs(scan['spacing'] - spacing)) > 1e-4:
                raise ValueError('Found non matching spacings for files\n'
                                 f'{path_to_image_file}\nand\n'
                                 f'{path_to_label_file}.')
            
        else:
            scan['spacing'] = spacing
            
        scan['label'] = lb
        scan['path_to_label_file'] = path_to_label_file
    
    if path_to_image_file is None and path_to_label_file is None:
        raise TypeError("No path was given as input for nifti reading.")
    
    return scan
        
class dataset(object):
    
    def __init__(self, path):
        
        self.path = path
        
        assert os.path.exists(self.path), f'path {self.path} does not exist'
        
        folders = [f for f in os.listdir(self.path) 
                   if os.path.isdir(os.path.join(self.path, f))]
        
        # identify image folder
        im_folders = [f for f in folders if f.startswith('images')]
        assert len(im_folders) == 1, f'Assumed to find one folder starting with \'images\' at {self.path}. Got {len(im_folders)} instead'
        self.image_folder = im_folders[0]
        
        # check for label folder
        lb_folders = [f for f in folders if f.startswith('labels')]
        assert len(lb_folders) < 2, f'Assumed to find no or one folder starting with \'labels\' at {self.path}. Got {len(lb_folders)} instead'
        self.label_folder = None if len(lb_folders) == 0 else lb_folders[0]
        
        # now get all nifti file names
        if self.label_folder:
            im_files = os.listdir(os.path.join(self.path, self.image_folder))
            lb_files = os.listdir(os.path.join(self.path, self.label_folder))
            
            missing_im_files = [f for f in lb_files if f not in im_files]
            
            if len(missing_im_files) > 0:
                print(f'The following files were missing in the image folder: {missing_im_files}')
            
            missing_lb_files = [f for f in im_files if f not in lb_files]
            
            if len(missing_lb_files) > 0:
                print(f'The following files were missing in the lb folder: {missing_lb_files}')
                
            if len(missing_im_files) + len(missing_lb_files) > 0:
                raise FileNotFoundError('Some files seems to missing in either the image or label folder (see above).')

        self.nii_file_list = os.listdir(os.path.join(self.path, self.image_folder))
    
    
    def __len__(self):
        return len(self.nii_file_list)
    
    def __iter__(self):
        self.counter = 0
        return self
    
    def __next__(self):
        
        if self.counter >= self.__len__():
            raise StopIteration
        
        self.counter += 1
        
        return self.__getitem__(self.counter-1)

    def __getitem__(self, index, image_only=False, label_only=False):
        
        nii_file = self.nii_file_list[index]
        path_to_image_file = os.path.join(self.path, self.image_folder, nii_file)
        path_to_label_file = os.path.join(self.path, self.label_folder, nii_file)
        
        if image_only:            
            return load_scan(path_to_image_file,
                             None)
        elif label_only:
            return load_scan(None,
                             path_to_label_file)
        else:
            return load_scan(path_to_image_file,
                             path_to_label_file)
        
    def get_median_spacing(self):
        print('computing median spacing...')
        
        spacings = []
        
        for nii_file in self.nii_file_list:
            path_to_file = os.path.join(self.path, self.image_folder, nii_file)
            nii_img = nib.load(path_to_file)
            spacings.append(nii_img.header['pixdim'][1:4])
        
        spacings = np.stack(spacings, 0)
        
        return np.median(spacings, 0)

def save_nii(arr, path_to_src_file, tar):
    
    # change orientation of the array back to nifti convention
    if arr.ndim == 3:    
        arr = np.moveaxis(arr[::-1, :, ::-1], 0, 2)
        arr = np.rot90(arr, 3, (0,1))
    elif arr.ndim == 4:
        arr = np.moveaxis(arr[:, ::-1, :, ::-1], 1, 3)
        arr = np.rot90(arr, 3, (1,2))
    else:
        raise ValueError(f"Array must be 3d or 4d to store as a nifti, but "
                         f"got {arr.ndim}d.")
    orig_nii_img = nib.load(path_to_src_file)
    new_nii_img = nib.Nifti1Image(arr,
                                  orig_nii_img.affine, 
                                  orig_nii_img.header)
    nib.save(new_nii_img, tar)