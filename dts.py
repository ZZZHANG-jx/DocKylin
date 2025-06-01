import torch 
from kmeans_pytorch import kmeans 
import math 
import cv2 
import numpy as np 




def dts(metric:torch.Tensor, metric2:torch.Tensor):
    t = metric.shape[1]  # (B,N,D)
    c = metric.shape[-1]

    metric = metric / metric.norm(dim=-1, keepdim=True)
    metric2 = metric2 / metric2.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        a, b = metric2[..., :, :], metric2[..., :, :]
        scores = a @ b.transpose(-1, -2)
        b, _, _ = scores.shape 
        scores_diag = torch.tril(torch.ones(t,t))*2
        scores_diag = scores_diag.expand(b, -1, -1).to(metric2.device)
        scores_diag = torch.eye(t).expand(b, -1, -1).to(metric2.device)*2
        scores = scores-scores_diag
        node_max, node_indx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        informative_index = edge_idx[..., t-50:, :]
        cluster_ids_x, cluster_centers = kmeans(X=metric[0], num_clusters=2, distance='cosine', device =metric2.device)
        labels = cluster_ids_x.to(metric2.device)

        unm_idx_0 = torch.where(labels==0)[0].view(1,-1,1)
        unm_idx_1 = torch.where(labels==1)[0].view(1,-1,1)

        within_labels_0_num = 0
        within_labels_1_num = 0 
        for index in informative_index[0,:,0]:
            if labels[index]==0:
                within_labels_0_num+=1
            else:
                within_labels_1_num+=1 
        if within_labels_0_num > within_labels_1_num:
            unm_idx = unm_idx_0
            m_idx = unm_idx_1
        else:
            unm_idx = unm_idx_1
            m_idx = unm_idx_0
    
    def indexing(src: torch.Tensor, token_index):
        n, t1, c = src.shape 
        r = token_index.shape[1]
        unm = src.gather(dim=-2,index=token_index.expand(n, r, c))
        token_index_new = token_index 
        all_idx = token_index_new 
        all_max, all_idx_idx = torch.sort(all_idx, dim=1)
        return unm.gather(dim=-2, index=all_idx_idx.expand(n, r, c))

    return indexing, unm_idx, m_idx, informative_index

def token_merge(x,y):
    def get_sim(x,y,eps=1e-6,mask_eye=-100,l2_norm=True):
        if y is None:
            y = x 
        if l2_norm:
            x = x / (x.norm(dim=-1, keepdim=True)+eps)
            y = y / (y.norm(dim=-1, keepdim=True)+eps)
        sim = torch.bmm(x, y.permute(0, 2, 1))
        if mask_eye is not None:
            sim.masked_fill_(
                torch.eye(x.size(1),device=x.device).unsqueeze(0).bool(),mask_eye)
        return sim 
    cos_sim = get_sim(y, x, mask_eye=None, l2_norm=True)
    cos_sim = cos_sim/1 
    sim_th = cos_sim.amax(dim=2, keepdims=True)
    mask = (cos_sim==sim_th).float()
    cos_sim = mask * cos_sim 
    mask = mask.permute(0, 2, 1)
    cos_sim = cos_sim.permute(0, 2, 1)
    numerator = torch.exp(cos_sim) * mask 
    denominator = math.e + numerator.sum(dim=-1, keepdims=True)
    x = x*(math.e / denominator)+torch.bmm(numerator/denominator,y)
    return x 

def ats_visualize(im:torch.Tensor, token_ids):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    im = im[0].permute(1,2,0).to('cpu').numpy()
    im = ((im*std+mean)*255).astype(np.uint8)
    im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    
    token_ids = token_ids.to('cpu').numpy()
    h,w = im.shape[:2]
    size = int(im.shape[1]/32)
    mask = np.zeros((int(im.shape[0]/32),int(im.shape[1]/32)))
    large_mask = np.zeros((h,w))
    for i in token_ids:
        y = i//size
        x = i%size
        large_mask[int(y*32):int((y+1)*32),int(x*32):int((x+1)*32)] = 1
    
    mask = large_mask
    mask = np.expand_dims(mask,-1)
    mask = np.tile(mask,(1,1,3))

    mask = 1-mask 
    masked_im = cv2.addWeighted(im,0.5,(mask*(240,176,0)).astype(np.uint8),0.5,0)
    cv2.imwrite('visualize.jpg',masked_im)



if __name__ == '__main__':


    image_encoder_output = torch.randn((1,3000,512))
    linear_projection_output = torch.randn((1,3000,512))

    indexing, essential_tokens, nonessential_tokens, tokens_informative = dts(image_encoder_output,linear_projection_output)

    # ats_visualize(image, essential_tokens[0:,0])

    essential_linear_projection_output = indexing(linear_projection_output, essential_tokens)
    nonessential_linear_projection_output = indexing(linear_projection_output, nonessential_tokens)

    inputs_qwen = token_merge(essential_linear_projection_output, nonessential_linear_projection_output)

