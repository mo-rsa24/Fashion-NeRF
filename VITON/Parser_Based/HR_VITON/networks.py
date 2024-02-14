import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import os
# from preprocessing.segment_anything.segment_anything import SamAutomaticMaskGenerator, SamPredictor
from torch.nn.utils import spectral_norm
from skimage.metrics import structural_similarity as ssim
from VITON.Parser_Based.HR_VITON.eval_models.evals_model import PerceptualLoss
from torchvision.models.inception import inception_v3
from torchvision.utils import save_image
import numpy as np
import cv2
import functools


tsrtonp = lambda tensor: tensor.detach().cpu().permute(1,2,0).numpy()

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        # Pointwise convolution
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Query, Key and Value projections
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        # Add the attention-weighted output to the input feature map
        out = out + x
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class ConditionGenerator(nn.Module):
    def __init__(self, opt, input1_nc, input2_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,segment_anything=None):
        super(ConditionGenerator, self).__init__()
        self.warp_feature = opt.warp_feature
        self.out_layer_opt = opt.out_layer
        self.opt = opt
        self.segment_anything = segment_anything
        self.ClothEncoder = nn.Sequential(
            ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  # 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),  # 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 8
        )
        
        self.PoseEncoder = nn.Sequential(
            ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )
        
        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')
        
        if opt.warp_feature == 'T1':
            # in_nc -> skip connection + T1, T2 channel
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 2 + ngf * 4 , ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.warp_feature == 'encoder':
            # in_nc -> [x, skip_connection, warped_cloth_encoder_feature(E1)]
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 3, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 3, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 3, ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 3, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.out_layer == 'relu':
            self.out_layer = ResBlock(ngf + input1_nc + input2_nc, output_nc, norm_layer=norm_layer, scale='same')
        if opt.out_layer == 'conv':
            self.out_layer = nn.Sequential(
                ResBlock(ngf + input1_nc + input2_nc, ngf, norm_layer=norm_layer, scale='same'),
                nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
            )
        
        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Person Conv 1x1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )
        
        self.flow_conv = nn.ModuleList([
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        )
        self.self_attention_modules = nn.ModuleList([
            SelfAttention(ngf),     # For features after first ResBlock
            SelfAttention(ngf * 2), # For features after second ResBlock
            SelfAttention(ngf * 4), # For features after third ResBlock
            SelfAttention(ngf * 4), # For features after fourth ResBlock
            SelfAttention(ngf * 4), # For features after fifth ResBlock
        ])
        
        
        self.flow_self_attention = nn.ModuleList([
            SelfAttention(ngf * 4),     # For features after first ResBlock
            SelfAttention(ngf * 4), # For features after second ResBlock
            SelfAttention(ngf * 4), # For features after third ResBlock
            SelfAttention(ngf * 4), # For features after fourth ResBlock
            SelfAttention(ngf * 4), # For features after fifth ResBlock
        ])
        
        
        # Initialize channel attention (SE) modules for different scales
        self.channel_attention = nn.ModuleList([
            ChannelAttention(ngf),     # For features after first ResBlock
            ChannelAttention(ngf * 2), # For features after second ResBlock
            ChannelAttention(ngf * 4), # For features after third ResBlock
            ChannelAttention(ngf * 4), # For features after fourth ResBlock
            ChannelAttention(ngf * 4), # For features after fifth ResBlock
        ])
        
        # Initialize channel attention (SE) modules for different scales
        self.flow_channel_attention = nn.ModuleList([
            ChannelAttention(ngf * 4),     # For features after first ResBlock
            ChannelAttention(ngf * 4), # For features after second ResBlock
            ChannelAttention(ngf * 4), # For features after third ResBlock
            ChannelAttention(ngf * 4), # For features after fourth ResBlock
            ChannelAttention(ngf * 4), # For features after fifth ResBlock
        ])
        
        # Initialize channel attention (SE) modules for different scales
        self.spatial_attention_blocks = nn.ModuleList([SpatialAttention() for _ in range(5)])
        self.bottleneck = nn.Sequential(
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True) , nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
        )
        
    def normalize(self, x):
        return x
    
    def forward(self,opt,input1, input2, upsample='bilinear', im_c=None):
        if im_c is not None and (opt.fine_width, opt.fine_height) == (768, 1024):
            im_c = transforms.Resize((256, 192))(im_c)
        E1_list = []
        E2_list = []
        flow_list = []
        # warped_grid_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E1 = self.ClothEncoder[i](input1)
                E2 = self.PoseEncoder[i](input2)
            else:
                E1 = self.ClothEncoder[i](E1_list[i - 1])
                E2 = self.PoseEncoder[i](E2_list[i - 1])
            
            # Apply self-attention on the encoded features before appending them to the list
            # This can be done at each scale or selectively at certain scales
            if self.opt.feature_pyramid_self_attention: 
                E1 = self.self_attention_modules[i](E1)
                E2 = self.self_attention_modules[i](E2)
            elif self.opt.feature_pyramid_spatial_attention:
                attention_map_E1 = self.spatial_attention_blocks[i](E1)
                attention_map_E2 = self.spatial_attention_blocks[i](E2)
                E1 = attention_map_E1 * E1
                E2 = attention_map_E2 * E2
            elif self.opt.feature_pyramid_channel_attention:
                E1 = self.channel_attention[i](E1)
                E2 = self.channel_attention[i](E2)
            E1_list.append(E1)
            E2_list.append(E2)
        
        # Compute Clothflow
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW,opt)

            if i == 0:
                T1 = E1_list[4 - i]  # (ngf * 4) x 8 x 6
                T2 = E2_list[4 - i]
                E4 = torch.cat([T1, T2], 1)
                
                flow = self.flow_conv[i](self.normalize(E4)).permute(0, 2, 3, 1)
                flow_list.append(flow)
                
                x = self.conv(T2)
                x = self.SegDecoder[i](x)
                
            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) 
                if self.opt.flow_self_attention:
                    T1 = self.flow_self_attention[i](T1) 
                elif self.opt.flow_channel_attention:
                    T1 = self.flow_channel_attention[i](T1) 
                elif self.opt.flow_spatial_attention:
                    T1 = self.spatial_attention_blocks[i](T1) * T1
                T1 += self.conv1[4 - i](E1_list[4 - i])
                
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) 
                if self.opt.flow_self_attention:
                    T2 = self.flow_self_attention[i](T2) 
                elif self.opt.flow_channel_attention:
                    T2 = self.flow_channel_attention[i](T2) 
                elif self.opt.flow_spatial_attention:
                    T2 = self.spatial_attention_blocks[i](T2) * T2
                T2 +=  self.conv2[4 - i](E2_list[4 - i]) 
                
                
                
                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)  # upsample n-1 flow
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')
                
                flow = flow + self.flow_conv[i](self.normalize(torch.cat([warped_T1, self.bottleneck[i-1](x)], 1))).permute(0, 2, 3, 1)  # F(n)
                flow_list.append(flow)

                if self.warp_feature == 'T1':
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4-i], warped_T1], 1))
                if self.warp_feature == 'encoder':
                    warped_E1 = F.grid_sample(E1_list[4-i], flow_norm + grid, padding_mode='border')
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4-i], warped_E1], 1))
        
 
        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW,opt)
        
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
        warped_input1 = F.grid_sample(input1, flow_norm + grid, padding_mode='border')
        
        x = self.out_layer(torch.cat([x, input2, warped_input1], 1))

        warped_c = warped_input1[:, :-1, :, :]
        warped_cm = warped_input1[:, -1:, :, :]
        
        if self.segment_anything is not None and im_c is not None:
            for i in range(len(warped_cm)):
                warped_cloth, warped_mask = warped_c[i, :,:,:], warped_cm[i, :,:,:]  # torch tensor
                nd_warped_cloth,nd_warped_mask = unnormalize(warped_cloth),unnormalize(warped_mask)
                sam_masks = get_warped_clothing_masks(nd_warped_cloth, self.segment_anything, verbose=False)
                sam_masks = [nd_warped_mask] + sam_masks
                torch_im_c = im_c[i, :, :, :]
                index, score  = get_best_evaluation(warped_cloth, torch_im_c,sam_masks, verbose=False)
                warped_cm[i] = get_tensor(sam_masks, index)
            
        return flow_list, x, warped_c, warped_cm
    
    
    
    
def get_warped_clothing_masks(nd_img, sam, verbose=False):
    if verbose:
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(nd_img)
        sorted_masks = sorted(masks, key=lambda segmentation:segmentation['area'],reverse=True)
        plot_bounding_boxes(nd_img, sorted_masks)
    best_masks = get_mask_using_points(nd_img, sam, verbose=verbose)
    return best_masks
    
def get_lpips():
    pass


def get_best_evaluation(warped_c, im_c, masks, verbose=False):
    ssim_mask_dict, lpips_dict, mse_dict = {},{},{}
    T4 = transforms.Compose([transforms.ToTensor(),transforms.Resize((128, 128)),transforms.Normalize(mean=(0.5),std=(0.5))])
    model = PerceptualLoss(model='net-lin',net='alex',use_gpu=False)
    model.eval()
    for i in range(len(masks)):
        warped_composite = composite_mask_and_warped_cloth(warped_c, masks[i]) # tensor
        if verbose:
            save_image(warped_composite, f'warped_composite_{i}.png')
            save_image(warped_c, f'warped_c_{i}.png')
        nd_warped_composite = unnormalize(warped_composite)
        nd_warped_composite = np.asarray(Image.fromarray(nd_warped_composite).convert("L"))
        nd_warped_im_c =  unnormalize(im_c)
        nd_warped_im_c = np.asarray(Image.fromarray(nd_warped_im_c).convert("L"))
        if verbose:
            cv2.imwrite(f"gray_warped_composite_{i}.png", nd_warped_composite)
            cv2.imwrite(f"gray_warped_im_c_{i}.png", nd_warped_im_c)
        ssim_mask_dict[i] = ssim(nd_warped_im_c, nd_warped_composite, data_range=255, gaussian_weights=True, use_sample_covariance=False)
        
        nd_warped_composite_LPIPS = T4(nd_warped_composite)
        nd_warped_im_c_LPIPS = T4(nd_warped_im_c)
        
        lpips_dict[i] = model.forward(nd_warped_composite_LPIPS, nd_warped_im_c_LPIPS).item()
        mse_dict[i] = F.mse_loss(warped_composite, im_c.detach().cpu())
            
    ssim_mask_dict = dict(sorted(ssim_mask_dict.items(), key=lambda item: item[1], reverse=True))
    lpips_dict = dict(sorted(lpips_dict.items(), key=lambda item: item[1]))
    mse_dict = dict(sorted(mse_dict.items(), key=lambda item: item[1]))
    index, score = list(ssim_mask_dict.items())[-1]
    lpips_index, lpips_score = list(lpips_dict.items())[0]
    return lpips_index, lpips_score



def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) for two binary masks.
    
    Parameters:
    mask1 (np.array): First binary mask.
    mask2 (np.array): Second binary mask.
    
    Returns:
    float: IoU score.
    """
    # Calculate Intersection
    intersection = np.logical_and(mask1, mask2).sum()

    # Calculate Union
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0

    return iou

def composite_mask_and_warped_cloth(warped_cloth_paired, warped_clothmask_paired):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
    clothmask = transform(warped_clothmask_paired)
    warped_cloth_paired = warped_cloth_paired.detach().cpu()
    composite = warped_cloth_paired * clothmask + torch.ones_like(warped_cloth_paired) * (1-clothmask)
    return composite

def get_tensor(masks, index):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
    warped_cm = masks[index]
    warped_cm = transform(warped_cm).cuda()
    return warped_cm

# This method will output masks that are helpful
def get_mask_using_points(nd_img, sam,height=256,width=192,verbose=False):
    points = np.array([[height//2, width//2]])
    label = np.array([1])
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(nd_img)
    masks, scores, logits = mask_predictor.predict(point_coords=points, point_labels=label, multimask_output=True)
    if verbose: # These verbose need to be saved as part of training images in /gluster/mmolefe so that we can track growth over time
        [cv2.imwrite(f'm_{i}.png', (masks[i] * 255).astype('uint8')) for i in range(3)]
    best_masks = [(masks[i] * 255).astype('uint8') for i in range(len(masks))]
    return best_masks

def plot_bounding_boxes(img, masks):
    # nd_img = unnormalize(img)
    plots = {}
    for idx, mask in enumerate(masks):
        x,y,w,h = mask['bbox']
        temp_img = img.copy()
        bounding_box_image = cv2.rectangle(temp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_box_title = f'draw_{idx}.png'
        cv2.imwrite(bounding_box_title, bounding_box_image)
        plots[bounding_box_title] = bounding_box_image
    return plots


def unnormalize(tensor, threshold=127, mean=(0.5,), std=(0.5,)):
    """
    Unnormalize an image given the mean and std. Works for both single-channel
    and three-channel images.
    
    Parameters:
    - tensor: a torch.Tensor image.
    - mean: the mean used for normalization.
    - std: the standard deviation used for normalization.
    
    Returns:
    - unnormalized_img: a NumPy array image in uint8 type.
    """
    # Check the number of channels in the tensor and adjust mean and std
    num_channels = tensor.shape[0]
    if num_channels == 3:
        mean = mean if len(mean) == 3 else (0.5, 0.5, 0.5)
        std = std if len(std) == 3 else (0.5, 0.5, 0.5)
    elif num_channels == 1:
        mean = mean if len(mean) == 1 else (0.5,)
        std = std if len(std) == 1 else (0.5,)
    else:
        raise ValueError("Tensor has unsupported number of channels: {}".format(num_channels))

    # Denormalize the image by reversing the normalization process
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    
    # Reshape mean and std to [C, 1, 1] to broadcast along channel dimension
    mean = mean[:, None, None]
    std = std[:, None, None]
    
    # Multiply by std and then add the mean
    tensor = tensor * std + mean

    # Clamp the tensor to ensure the values are in the valid range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to [0, 255] and change to uint8
    tensor = tensor.mul(255).to(torch.uint8)
    
    # Convert to NumPy array and transpose to get (Height, Width, Channels)
    unnormalized_img = tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Ensure the array is contiguous in memory
    unnormalized_img = np.ascontiguousarray(unnormalized_img)
    if num_channels == 1:
        unnormalized_img = np.where(unnormalized_img > threshold, 255, 0).astype(np.uint8)[:,:,0]
    return unnormalized_img


def calculate_fit_tightness(img):
    
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray_img, 100, 200)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour assuming it is the object of interest
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Create a mask for the bounding rectangle
        mask = np.zeros_like(gray_img)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        # Measure how much of the bounding box is filled by the contour
        filled = np.sum(mask & edges) / np.sum(mask)
        return filled
    else:
        return 0

def make_grid(N, iH, iW,opt):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    if opt.cuda:
        grid = torch.cat([grid_x, grid_y], 3).cuda()
    else:
        grid = torch.cat([grid_x, grid_y], 3)
    return grid


class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_nc, out_nc, kernel_size=1,bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
            
        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    

class VGGLoss(nn.Module):
    def __init__(self, opt,layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False, Ddownx2=False, Ddropout=False, spectral=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.Ddownx2 = Ddownx2


        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, Ddropout, spectral=spectral)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        
        result = []
        if self.Ddownx2:
            input_downsampled = self.downsample(input)
        else:
            input_downsampled = input
        for i in range(num_D):
            
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, Ddropout=False, spectral=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm if spectral else lambda x: x

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            if Ddropout:
                sequence += [[
                self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                norm_layer(nf), nn.LeakyReLU(0.2, True), nn.Dropout(0.5)
            ]]
            else:

                sequence += [[
                    self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]
                
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


def save_checkpoint(model, save_path,opt):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    if opt.cuda:
        model.cuda()

def load_checkpoint(model, checkpoint_path,opt=None):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    if opt is None:
        model.cuda()
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D(input_nc, ndf=64, n_layers_D=3, norm='instance', use_sigmoid=False, num_D=2, getIntermFeat=False, gpu_ids=[], Ddownx2=False, Ddropout=False, spectral=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat, Ddownx2, Ddropout, spectral=spectral)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        # netD.cuda()
    netD.apply(weights_init)
    return netD
