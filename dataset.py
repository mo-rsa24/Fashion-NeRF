#coding=utf-8
import random
import sys
import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from glob import glob
import os.path as osp
import numpy as np
import json
# from NeRF.Vanilla_NeRF.load_blender import pose_spherical
from utils import get_transforms_data, get_transform_matrix, similarity, labels
class FashionNeRFDataset(data.Dataset):
    """
        Test Dataset for CP-VTON.
        models = ['nerf', 'viton', 'dm-vton', 'ladi-vton']
    """

    def __init__(self, root_opt,opt, viton=True, mode='train', model="NeRF",single_person_clothing=None, downsize=False):
        super(FashionNeRFDataset, self).__init__()
        # base setting
        self.opt = root_opt
        self.opt = opt 
        self.semantic_nc = opt.semantic_nc
        self.viton = viton
        self.model = model
        self.mode = mode
        self.person = root_opt.person
        self.downsize = downsize
        self.single_person_clothing = single_person_clothing
        self.clothing = root_opt.clothing
        self.in_shop_clothing = root_opt.in_shop_clothing
        self.person_clothing = f"{self.person}_{self.clothing}"
        if mode == 'inference':
            self.full_input_data_path = osp.join(root_opt.dataroot,'inference')
            self.data_path = osp.join(self.full_input_data_path, self.person_clothing)
            data_items = glob(f"{self.data_path}/image/*")
        elif model == 'viton':
            self.full_input_data_path = os.path.join(root_opt.root_dir, root_opt.rail_dir)
            if mode == "test":
                self.full_input_data_path = self.full_input_data_path.replace("train", mode)
            if mode == "evaluate":
                self.full_input_data_path = self.full_input_data_path.replace("evaluate", "test")
            self.data_path = self.full_input_data_path
            data_items = glob(f"{self.full_input_data_path}/image/*")
        elif model == 'NeRF':
            self.full_input_data_path = osp.join(root_opt.dataroot, mode)
            self.data_path = self.full_input_data_path
            data_items = glob(f"{self.full_input_data_path}/image/*{self.person}*")
        if root_opt.dataset_name == 'Rail':
            self.transforms_dir = osp.join(root_opt.root_dir, root_opt.transforms_dir)
            self.cihp = "./cihp_pgn.sh"
            self.detectron = "./densepose.sh"
            self.openpose = "./openpose.sh"
            self.mode = mode
            self.parse_agnostic = "./parse_agnostic.sh"
            self.fine_height = opt.fine_height
            self.fine_width = opt.fine_width
            self.semantic_nc = opt.semantic_nc
            self.transforms_data = get_transforms_data(self.transforms_dir, self.person_clothing)
            if self.transforms_data is not None:
                self.camera_angle_x = self.transforms_data['camera_angle_x']
                self.camera_angle_y = self.transforms_data['camera_angle_y']
                self.fl_x = self.transforms_data['fl_x']
                self.fl_y = self.transforms_data['fl_y']
                self.k1 = self.transforms_data['k1']
                self.k2 = self.transforms_data['k2']
                self.k3 = self.transforms_data['k3']
                self.k4 = self.transforms_data['k4']
                self.p1 = self.transforms_data['p1']
                self.p2 = self.transforms_data['p2']
                self.is_fisheye = self.transforms_data['is_fisheye']
                self.cx = self.transforms_data['cx']
                self.cy = self.transforms_data['cy']
                self.W = self.transforms_data['w']
                self.H = self.transforms_data['h']
                self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)
                self.K = np.array([[self.focal, 0, 0.5 * self.W], [0, self.focal, 0.5 * self.H], [0, 0, 1]])
                self.aabb_scale = self.transforms_data['aabb_scale']
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        im_names = []
        transform_matrices =[]
        get_clothing_name = "_"
        self.transform_wo_normalize = transforms.Compose([transforms.ToTensor()])
        if root_opt.dataset_name == 'Rail':
            for data_item in data_items:
                data_string = data_item.split("/")[-1]
                if model == 'NeRF':
                    frames = self.transforms_data['frames']
                    frame = next(filter(lambda file_path: data_string in (file_path.get('file_path')), frames), None)
                    transform_matrices.append(frame['transform_matrix'])
        else:
            if self.mode == 'inference':
                im_names = [self.single_person_clothing['person_filename']]
        
        self.c_names = dict()
        
        if root_opt.dataset_name ==  'Rail':
            self.get_clothing_name = lambda path_to_image:"_".join(path_to_image.split("/")[-1].split("_")[:-1])
            self.clothing_names = [f"{self.get_clothing_name(image)}.jpg" for image in data_items]
            im_names = [image.split("/")[-1] for image in data_items]
            self.c_names['paired'] = list(set(self.clothing_names))
            self.c_names['unpaired'] = random.sample(self.c_names['paired'], len(self.c_names['paired']))
        else:
            c_names = []
            with open(osp.join(self.full_input_data_path, f"{mode}_pairs.txt"), 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_names.append(im_name)
                    c_names.append(c_name)
            
            self.c_names['paired'] = im_names
            self.c_names['unpaired'] = c_names
        self.im_names = im_names
        if root_opt.dataset_name == 'Rail':
            self.transform_matrices = transform_matrices if self.model == 'NeRF' else None
            # self.render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-60,60,40+1)[:-1]], 0)

    def name(self):
        return "FashionNeRFData"

    def remove_artifacts(self, image, threshold=0.92):
        img = image.copy()
        x, y = image.shape[:2]
        for i in range(x):
            for j in range(y):
                if self.isRemovable(img[i, j], threshold):
                    img[i, j] = np.array([0, 0, 0])
        return img


    def isRemovable(self, pixel, threshold=0.92):
        avg = np.average(pixel)
        boolean = True
        for value in pixel:
            if similarity(value, avg) < threshold:
                boolean = False
        return boolean
    
    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # mask_arm = Image.new('L', (self.opt.fine_width, self.fine_height), 'white')
            mask_arm = Image.new('L', (768, 1024), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'black',
                                          'black')
            mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black', 'black')

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic
    
    
    def get_cihp_data(self, im_name):
        filename = im_name.split("/")[-1]
        cihp_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')            
        if not os.path.isfile(os.path.join(self.data_path, cihp_name)):
            self.cihp += f" {self.full_input_data_path} {self.person} {self.clothing} {filename}"
            cihp_err = os.system(self.cihp)
            if cihp_err:
                print("FATAL: CIHP command failed")
                sys.exit(cihp_err)
        # parse_name = im_name.replace('image', 'cihp').replace('.jpg', '.png')  # VITON
        im_parse_pil_big = Image.open(os.path.join(self.data_path, cihp_name))
        im_parse = transforms.Resize(self.opt.fine_width, interpolation=0)(im_parse_pil_big)
        return im_parse,im_parse_pil_big
        
    def get_agnostic_parse_data(self, im_name):
        filename = im_name.split("/")[-1]
        parse_agnostic_name = im_name.replace('image', 'image-parse-agnostic-v3.2').replace('.jpg', '.png')
        if not os.path.isfile(os.path.join(self.data_path, parse_agnostic_name)):
            self.parse_agnostic += f" {self.full_input_data_path} {self.person} {self.clothing} {filename}"
            parse_agnostic_err = os.system(self.parse_agnostic)
            if parse_agnostic_err:
                print("FATAL: Parse agnostic command failed")
                sys.exit(parse_agnostic_err)

        image_parse_agnostic = Image.open(os.path.join(self.data_path, parse_agnostic_name))
        image_parse_agnostic = transforms.Resize(self.opt.fine_width, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))
        parse_agnostic_map = torch.FloatTensor(20, self.opt.fine_height, self.opt.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.opt.fine_height, self.opt.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        return parse_agnostic_map, new_parse_agnostic_map
    
    def get_parse_data(self, im_parse):
        parse = torch.from_numpy(np.array(im_parse)[None]).long()
        parse_map = torch.FloatTensor(20, self.opt.fine_height, self.opt.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.opt.fine_height, self.opt.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.opt.fine_height, self.opt.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i
        return parse, new_parse_map, parse_onehot
        
    def get_openpose_data(self, im_name):
        filename = im_name.split("/")[-1]
        openpose_name = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')    
        if not os.path.isfile(os.path.join(self.data_path, openpose_name)):
            self.openpose += f" {self.full_input_data_path} {self.person} {self.clothing} {filename}"
            openpose_err = os.system(self.openpose)
            if openpose_err:
                print("FATAL: Openpose command failed")
                sys.exit(1)


        pose_map = Image.open(os.path.join(self.data_path, openpose_name))
        pose_map = transforms.Resize(self.opt.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]

        pose_name = im_name.replace('image', 'openpose_json').replace('.jpg','_keypoints.json')  # VITON
        with open(os.path.join(self.data_path, pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
        return pose_map, pose_data
    
    def get_densepose_data(self, im_name):
        filename = im_name.split("/")[-1]
        densepose_name = im_name.replace('image', 'image-densepose')
        if self.mode != 'test' and self.mode != 'train':
            self.detectron += f" {self.full_input_data_path} {self.person} {self.clothing} {filename}"
            densepose_err = os.system(self.detectron)
            if densepose_err:
                print("FATAL: Densepose command failed")
                sys.exit(1)

        densepose_map = Image.open(os.path.join(self.data_path, densepose_name))
        densepose_map = transforms.Resize(self.opt.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)  # [-1,1]
    
        densepose_map_wo_normalize = self.transform_wo_normalize(Image.open(os.path.join(self.data_path, densepose_name)))
        densepose_end_of_torso_mask = torch.FloatTensor((densepose_map_wo_normalize[1:2,:,:].cpu().numpy() == (80/ 255.)).astype(int)) # a channel of torso region has 80 value in a painter app.
        densepose_end_of_torso_mask = transforms.Resize(self.fine_width, interpolation=0)(densepose_end_of_torso_mask)
        return densepose_map, densepose_map_wo_normalize, densepose_end_of_torso_mask
    
    def get_agnostic_data(self, im_pil_big, im_parse_pil_big, pose_data):
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(self.opt.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)
        return agnostic
    
    
    def make_grid(self, N, iH, iW):
        grid_x = torch.linspace(0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
        grid_y = torch.linspace(0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
        grid = torch.cat([grid_x, grid_y], 3)
        return grid    
    
    def __getitem__(self, index):
        if self.single_person_clothing is None:
            im_name = self.im_names[index]
            im_name = "image/" + im_name
            if self.opt.dataset_name == 'Rail':
                transform_matrix = np.array(self.transform_matrices[index]) if self.transform_matrices is not None else None
                transform_matrix = torch.tensor(transform_matrix) if transform_matrix is not None else None
                focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)
                K = np.array([[focal, 0, 0.5 * self.W], [0, focal, 0.5 * self.H], [0, 0, 1]])
            filename = im_name.split("/")[-1]
            im_pil_big = Image.open(osp.join(self.data_path, im_name))
            if self.model != 'NeRF':
                c_name = {}
                c = {}
                cm = {}
                for key in self.c_names:
                    if self.opt.dataset_name == 'Rail':
                        if key == 'paired':
                            clothing_index = self.c_names[key].index(f"{self.get_clothing_name(im_name)}.jpg")
                        else:
                            clothing_index = random.randint(0, len(self.c_names[key])-1)
                        # print(f"Careful: {self.c_names[key][clothing_index]}")
                        c_name[key] = self.c_names[key][clothing_index]
                    else:
                        c_name[key] = self.c_names[key][index]    
                    c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
                    c[key] = transforms.Resize(self.opt.fine_width, interpolation=2)(c[key])
                    cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
                    cm[key] = transforms.Resize(self.opt.fine_width, interpolation=0)(cm[key])

                    c[key] = self.transform(c[key])  # [-1,1]
                    cm_array = np.array(cm[key])
                    cm_array = (cm_array >= 128).astype(np.float32)
                    cm[key] = torch.from_numpy(cm_array)  # [0,1]
                    cm[key].unsqueeze_(0)

                im_pil = transforms.Resize(self.opt.fine_width, interpolation=2)(im_pil_big)
                im = self.transform(im_pil.convert('RGB'))
                result = {
                'c_name': c_name, 'cloth': c, 'cloth_mask': cm,  # for input
                'im_name': im_name,  'image': im}
            else:
                if self.downsize == False:
                    im = self.transform(im_pil_big.convert('RGB'))
                    result = {'im_name': im_name,  'image': im,'H': self.H,'W': self.W,'K': K}
                else:
                    im_pil = transforms.Resize((256, 192), interpolation=2)(im_pil_big)
                    im = self.transform(im_pil.convert('RGB'))
                    result = {'im_name': im_name, 'image': im, 'H': self.H, 'W': self.W, 'K': K}
            if self.model == 'NeRF':
                result.update({'transform_matrix': transform_matrix,'H': self.H,'W': self.W,'K': K})


            if self.viton:
                im_parse,im_parse_pil_big = self.get_cihp_data(im_name)
                parse, new_parse_map, parse_onehot = self.get_parse_data(im_parse)
                pose_map, pose_data = self.get_openpose_data(im_name)
                parse_agnostic_map, new_parse_agnostic_map = self.get_agnostic_parse_data(im_name)    
                densepose_map, densepose_map_wo_normalize, densepose_end_of_torso_mask = self.get_densepose_data(im_name)
                pcm = new_parse_map[3:4]
                im_c = im * pcm + (1 - pcm) # Extract clothing Sc
                agnostic = self.get_agnostic_data(im_pil_big, im_parse_pil_big, pose_data)
                lower_clothes_mask = new_parse_map[4:5,:,:]

                grid = self.make_grid(1, self.fine_height, self.fine_width).permute(0, 3, 1, 2)
                grid_x, grid_y = torch.split(grid, 1, dim=1)
                grid_y_max = (1. - densepose_end_of_torso_mask) * 0. + grid_y * densepose_end_of_torso_mask
                grid_y_max = torch.max(grid_y_max)
                grid_y_max_idx = grid_y_max * self.fine_height
                grid_y_max_idx = int(grid_y_max_idx)

                clothes_no_loss_mask = torch.zeros_like(densepose_end_of_torso_mask)
                clothes_no_loss_mask[:, :grid_y_max_idx, :] = 1
                result.update({
                    'im_name':im_name.replace("image/",""),'agnostic': agnostic, 'parse_agnostic': new_parse_agnostic_map, 'densepose': densepose_map, 'pose': pose_map,  # for conditioning
                    'parse_onehot': parse_onehot, 'parse': new_parse_map, 'pcm': pcm, 'parse_cloth': im_c,'image': im, 
                    # masks for a masked loss
                    'lower_clothes_mask': lower_clothes_mask,
                    'clothes_no_loss_mask': clothes_no_loss_mask
                    })

            return result
        else:
            person = self.single_person_clothing['person']
            person_filename = self.single_person_clothing['person_filename']
            clothing = self.single_person_clothing['clothing']
            clothing_filename = self.single_person_clothing['clothing_filename']
            clothing_mask = Image.open(clothing_filename.replace("cloth","cloth-mask"))
            im_parse,im_parse_pil_big = self.get_cihp_data(person_filename)
            parse, new_parse_map, parse_onehot = self.get_parse_data(im_parse)
            pose_map, pose_data = self.get_openpose_data(person_filename)
            parse_agnostic_map, new_parse_agnostic_map = self.get_agnostic_parse_data(person_filename)    
            densepose_map, densepose_map_wo_normalize, densepose_end_of_torso_mask = self.get_densepose_data(person_filename, remove_artifacts = True)
            pcm = new_parse_map[3:4]
            im_c = im * pcm + (1 - pcm) # Extract clothing Sc
            agnostic = self.get_agnostic_data(im_pil_big, im_parse_pil_big, pose_data)
            # masks for a masked loss
            lower_clothes_mask = new_parse_map[4:5,:,:]

            grid = self.make_grid(1, self.fine_height, self.fine_width).permute(0, 3, 1, 2)
            grid_x, grid_y = torch.split(grid, 1, dim=1)
            grid_y_max = (1. - densepose_end_of_torso_mask) * 0. + grid_y * densepose_end_of_torso_mask
            grid_y_max = torch.max(grid_y_max)
            grid_y_max_idx = grid_y_max * self.fine_height
            grid_y_max_idx = int(grid_y_max_idx)

            clothes_no_loss_mask = torch.zeros_like(densepose_end_of_torso_mask)
            clothes_no_loss_mask[:, :grid_y_max_idx, :] = 1
            result = {
                'c_name': clothing_filename, 'cloth': clothing, 'cloth_mask': clothing_mask,  # for input
                'im_name': person_filename,  'image': person,'H': self.H,'W': self.W,'K': K,
                'agnostic': agnostic, 'parse_agnostic': new_parse_agnostic_map, 
                'densepose': densepose_map,  'pose': pose_map, 'parse_onehot': parse_onehot,
                'parse': new_parse_map, 'pcm': pcm, 'parse_cloth': im_c,'lower_clothes_mask': lower_clothes_mask,
                    'clothes_no_loss_mask': clothes_no_loss_mask
                }
            return result

    def __len__(self):
        return len(self.im_names)


class FashionDataLoader(object):
    def __init__(self,dataset, batch_size, workers, shuffle):
        super(FashionDataLoader, self).__init__()

        if shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers,  drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
