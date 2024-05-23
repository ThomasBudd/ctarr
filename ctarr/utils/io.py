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

def load_nii_file(nii_file):
    if not os.path.exists(nii_file):
        raise FileNotFoundError(f'Couldn\'t read {nii_file}, file does not exsist.')
    
    img = nib.load(nii_file)
    spacing = img.header['pixdim'][1:4]
    im = img.get_fdata()
    
    assert im.ndim in [3, 4], "Image from nifti file must be 3d"
    
    # add empty channels axis
    if im.ndim == 3:
        im = im[None]
    
    return im, spacing

def load_scan(path,
              file_name,
              image_folder=None,
              label_folder=None):
    
    scan = {}
    
    if image_folder:
        path_to_image_file = os.path.join(path, image_folder, file_name)
        im, spacing = load_nii_file(path_to_image_file)
        
        scan['image'] = im
        scan['spacing'] = spacing
        scan['path_to_image_file'] = path_to_image_file
        scan['orig_shape'] = im.shape[-3:]
    
    if label_folder:
        
        path_to_label_file = os.path.join(path, label_folder, file_name)
        lb, spacing = load_nii_file(path_to_label_file)
        
        orig_shape = lb.shape[-3:]
        if 'spacing' in scan:
            if np.max(np.abs(scan['spacing'] - spacing)) > 1e-4:
                raise ValueError(f'Found non matching spacings for file {file_name}')
            
            if orig_shape != scan['orig_shape']:
                raise ValueError(f'Found non matching shaped for file {file_name}')     
            
        else:
            scan['spacing'] = spacing
            scan['orig_shape'] = orig_shape
            
        scan['label'] = lb
        scan['path_to_label_file'] = path_to_label_file
        scan['orig_shape'] = lb.shape[-3:]
    
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
        
        if image_only:            
            return load_scan(self.path,
                             self.nii_file_list[index], 
                             self.image_folder,
                             None)
        elif label_only:
            return load_scan(self.path,
                             self.nii_file_list[index], 
                             None,
                             self.label_folder)
        else:
            return load_scan(self.path,
                             self.nii_file_list[index], 
                             self.image_folder,
                             self.label_folder)
        
    def get_median_spacing(self):
        print('computing median spacing...')
        
        spacings = []
        
        for nii_file in self.nii_file_list:
            path_to_file = os.path.join(self.path, self.image_folder, nii_file)
            nii_img = nib.load(path_to_file)
            spacings.append(nii_img.header['pixdim'][1:4])
        
        spacings = np.stack(spacings, 0)
        
        return np.median(spacings, 0)

def save_nii(arr, src, tar):
    
    orig_nii_img = nib.load(src)
    new_nii_img = nib.Nifti1Image(arr, orig_nii_img.affine, orig_nii_img.header)
    nib.save(new_nii_img, tar)