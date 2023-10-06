""" One-Shot Affordance Correspondence Model - AffCorrs """
# Standard imports
import gc
import numpy as np
from PIL import Image
# Skimage
from skimage.transform import resize
from sklearn.decomposition import PCA
from skimage import img_as_bool
from sklearn.manifold import Isomap
# Pytorch
import torch
from pytorch_metric_learning import distances
# Custom imports
from .extractor import ViTExtractor
from .crf import CRF
from .clustering import get_K_means_v2
from .correspondence_functions import (get_load_shape,
                                       image_resize,
                                       get_saliency_maps_V2,
                                       map_descriptors_to_clusters,
                                       rescale_pts,
                                       uv_im_to_desc,
                                       uv_desc_to_im,
                                       map_values_to_segments,
                                       )
import matplotlib.pyplot as plt

def to_int(x):
  """ Returns an integer type object 
  : param x: torch.Tensor, np.ndarray or scalar. 
  """
  if torch.is_tensor(x): return x.long()
  if isinstance(x, np.ndarray): return x.astype(np.int32)
  return int(x)
  
class AffCorrs_V1(object):
    """ AffCorrs model - Version 1
    The model is capable of finding one-shot region correspondences
    with no fine-tuning.
    """
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ViTExtractor(self.args['model_type'], 
                                      self.args['stride'], 
                                      device=self.device)
        self.d_fn = distances.CosineSimilarity()
        # TODO: Compare Layer 9 vs Layer 11 descriptors. Note that this
        # is only used for point correspondence which is optional.
        self.use_L9_corrs = True

        self.all_feats = [] # This variable is used to store the features of both source and destination
        return

    def _cleanup(self):
        """ Private cleanup action """
        torch.cuda.empty_cache()
        gc.collect() 

    def set_source(self, src_im, parts, kps=None):
        """ Set the source image and parts.
        :param src_im: path to, or PIL.Image
        :param parts: List[np.ndarray], list of masks for each part
        :param kps: [Optional] List[(2,1)], list of keypoints to find
                    the point correspondences of. 
        """
        self._cleanup()
        if isinstance(src_im, str):
          src_im = Image.open(src_im)
        if not isinstance(src_im, Image.Image):
          raise TypeError("Wrong type of input im - must be PIL.Image")
        self.src_img = src_im
        self.src_query_parts = parts
        self.src_query_kps  = kps
        self.preprocess_source()
        return 
      
    def preprocess_source(self):
        """ Prepares source descriptors, clusters, etc.
        """
        with torch.no_grad():
            load_size = self.args["load_size"]
            self.src_load_shape = get_load_shape(self.src_img, load_size) 
            self.src_img_resized = image_resize(self.src_img,[load_size,None])
            self.src_query_parts  = [img_as_bool(resize(p, self.src_load_shape)) \
                          for p in self.src_query_parts]
            self.src_img_batch,_ = self.model.preprocess_image(
                                        self.src_img_resized, self.src_load_shape)
            self.src_descs = self.model.extract_descriptors(
                                        self.src_img_batch.to(self.device), 
                                        self.args["layer"], 
                                        self.args["facet"], 
                                        self.args["bin"]).cpu()
            self.src_num_patches = self.model.num_patches
            if self.use_L9_corrs:
                self.src_corr_descs = self.model.extract_descriptors(
                                            self.src_img_batch.to(self.device), 
                                            layer=9, facet="key", bin=True
                                            ).cpu().detach()
            else: 
                self.src_corr_descs = self.src_descs
            self.src_corr_num_patches = self.model.num_patches
            from sklearn.manifold import Isomap
            for i in range(len(self.model._feats)):
                print(self.model._feats[i].shape)
            
        return

    def generate_source_clusters(self, N_Kq=10):
        """ Generate source clusters 
        :param N_Kq: parameter for number of query clusters
        """
        self.N_Kq = N_Kq
        nps = self.src_num_patches
        self.src_desc_image = self.src_descs.numpy().reshape(nps[0],nps[1],-1)
        print(self.src_desc_image.shape, self.src_desc_image.dtype, self.src_desc_image.max(), self.src_desc_image.min())
        # exit()
        self.K_queries = []
        for part in self.src_query_parts:
            q_a = resize(part, nps).astype(bool)
            # extract Source K-means descriptors
            query_desc = self.src_desc_image[q_a]
            K_query, labels = get_K_means_v2([query_desc[None,None,...]], 1, 
                                            self.args["elbow"], 
                                            list(range(N_Kq,N_Kq+1)))
            plt.imshow(q_a)
            plt.show()
            # print((q_a == True).sum())
            colors = np.random.rand(10, 3)
            cluster_image = np.zeros((*nps, 3))
            cluster_image[q_a==True] = colors[labels.cpu().numpy()]
            cluster_image = resize(cluster_image, self.src_img_resized.size[::-1])
            cluster_image = (cluster_image * 255)*0.5 + 0.5*np.asarray(self.src_img_resized)
            cluster_image = cluster_image.astype(np.uint8)
            plt.imshow(cluster_image)
            plt.show()
            print(K_query.shape, len(labels), q_a.shape)
            # exit()
            self.K_queries.append(K_query)


    def set_target(self, tgt_im, tgt_mask=None):
        """ Set the source image and parts.
        :param tgt_im: path to, or PIL.Image
        :param tgt_mask: [Optional] np.ndarray, Foreground mask. If None
                          is passed, the mask is based on saliency map.
        """
        self._cleanup()
        if isinstance(tgt_im, str):
            tgt_im = Image.open(tgt_im)
        if not isinstance(tgt_im, Image.Image):
            raise TypeError("Wrong type of input im - must be PIL.Image")

        self.tgt_img = tgt_im
        self.tgt_mask = tgt_mask
        self.preprocess_target()
        return

    def preprocess_target(self):
        """ Prepares target descriptors, clusters, etc.
        """
        with torch.no_grad():
            load_size = self.args["load_size"]
            tgt_load_shape = get_load_shape(self.tgt_img, load_size) 
            self.tgt_img_resized = image_resize(self.tgt_img,[load_size,None])
            self.tgt_img_batch,_ = self.model.preprocess_image(
                                        self.tgt_img_resized, tgt_load_shape)
            self.tgt_descs = self.model.extract_descriptors(
                                        self.tgt_img_batch.to(self.device), 
                                        self.args["layer"], 
                                        self.args["facet"], 
                                        self.args["bin"]).cpu()
            print(self.model._feats[0].shape, self.model._feats[0].dtype, self.model._feats[0].max(), self.model._feats[0].min())
            print(self.model.num_patches)
            from sklearn.manifold import Isomap


            self.tgt_num_patches = self.model.num_patches
            self.tgt_load_size = self.model.load_size
            if self.use_L9_corrs:
                self.tgt_corr_descs = self.model.extract_descriptors(
                                          self.tgt_img_batch.to(self.device), 
                                          layer=9, facet="key", bin=True
                                          ).cpu().detach() 
            else:
                self.tgt_corr_descs = self.tgt_descs
            self.tgt_corr_num_patches = self.model.num_patches
            self.get_saliency_mask()
        return
    
    def get_saliency_mask(self):
        """ Get saliency mask """
        # Get Target mask
        if self.tgt_mask is not None:
            saliency_map = img_as_bool(resize(self.tgt_mask, 
                                            self.tgt_num_patches))  
        else:
            print("Generating Saliency mask")
            saliency_map = get_saliency_maps_V2(self.model, self.tgt_img_batch, 
                              self.tgt_img, self.tgt_num_patches, 
                              self.tgt_load_size, 
                              self.args["low_res_saliency_maps"], 
                              self.device, thresh=0.065 )
        self.tgt_saliency_mask = saliency_map
        print(saliency_map.shape, saliency_map.dtype, saliency_map.max(), saliency_map.min())
        plt.imshow(saliency_map)
        plt.title("Saliency map")
        plt.show()
        plt.imshow(self.tgt_img)
        plt.title("Target image")
        plt.show()

        import cv2
        saliency_map = np.asarray(saliency_map, dtype=np.float32)
        print(saliency_map.shape, saliency_map.dtype)
        tgt_img = np.asarray(self.tgt_img, dtype=np.float32)
        saliency_map = cv2.resize(saliency_map, tgt_img.shape[:2][::-1])
        saliency_map[saliency_map<0.5] = 0.5
        masked_img = saliency_map[...,None] * self.tgt_img
        plt.imshow(masked_img.astype(np.uint8))
        plt.title("Target image masked")
        plt.show()
        # exit()
        return

    
    def generate_target_clusters(self, N_Kt=80, N_Kt_bg=80):
        """ Generate target clusters 
        :param N_Kt: parameter for number of target clusters
        :param N_Kt_bg: parameter for number of target background clusters
        """
        nps = self.tgt_num_patches
        self.tgt_desc_image = self.tgt_descs.numpy().reshape(nps[0],nps[1],-1)
        fg_desc = self.tgt_desc_image [self.tgt_saliency_mask==True]
        K_target, labels = get_K_means_v2([fg_desc[None,None,...]], 1, 
                                          self.args["elbow"], 
                                          list(range(N_Kt,N_Kt+1)))
        # plt.imshow(q_a)
        # plt.show()
        # print((q_a == True).sum())
        print(K_target)
        colors = np.random.rand(len(K_target), 3)
        cluster_image = np.zeros((*nps, 3))
        cluster_image[self.tgt_saliency_mask==True] = colors[labels.cpu().numpy()]
        cluster_image = resize(cluster_image, self.tgt_img_resized.size[::-1])
        cluster_image = (cluster_image * 255)*0.5 + 0.5*np.asarray(self.tgt_img_resized)
        cluster_image = cluster_image.astype(np.uint8)
        plt.imshow(cluster_image)
        plt.show()
        
        bg_desc = self.tgt_desc_image [self.tgt_saliency_mask==False]
        K_target_bg, labels = get_K_means_v2([bg_desc[None,None,...]], 8, 
                                                self.args["elbow"],
                                                list(range(N_Kt_bg,N_Kt_bg+1)))
        
        C = map_descriptors_to_clusters(self.tgt_descs[0,0], K_target, K_target_bg)
        C = C.reshape(nps[0], nps[1])

        self.N_Kt = N_Kt
        self.N_Kt_bg = N_Kt_bg
        self.K_target = K_target 
        self.K_target_bg = K_target_bg
        self.K_target_mapped = C
        colors = np.random.rand(len(K_target_bg), 3)
        cluster_image = np.zeros((*nps, 3))
        cluster_image[self.tgt_saliency_mask==False] = colors[labels.cpu().numpy()]
        cluster_image = resize(cluster_image, self.tgt_img_resized.size[::-1])
        cluster_image = (cluster_image * 255)*0.5 + 0.5*np.asarray(self.tgt_img_resized)
        cluster_image = cluster_image.astype(np.uint8)
        plt.imshow(cluster_image)
        plt.show()
        return

    def find_part_correspondences(self, temp_ts=0.02, temp_qt=0.2):
        """ Find part correspondences
        : param temp_ts: float, Target to support temperature
        : param temp_qt: float, Query to target temperature
        """
        tgt_nps = self.tgt_num_patches
        src_nps = self.src_num_patches
        # Mapping from Target centroids to Source descriptors
        A_ts = self.d_fn(self.src_descs[0,0], 
                        torch.Tensor(self.K_target)).T # K2, H1.W1

        # Transform to probability: Softmax along HW axis (HW,K)
        # Probability of each cluster belonging a particular area
        # of the source image.
        P_ts = torch.nn.Softmax(dim=1)(A_ts/temp_ts) # [K2, H1.W2]
        P_ts_img = P_ts.reshape(-1,src_nps[0], src_nps[1]) # [K2, H1, W2]
        I2 = torch.ones(tgt_nps)*(0.5*self.N_Kq/self.N_Kt) # 0.0625

        # Output is [N parts, Target Size]
        load_shape = self.tgt_img_batch[0,0].shape
        segm_out = np.zeros((len(self.src_query_parts), *load_shape))
        for i, part in enumerate(self.src_query_parts):
            # Get similarity from K1 to K2
            A_qt = self.d_fn(torch.Tensor(self.K_queries[i]), 
                      torch.Tensor(self.K_target)) # K1,K2
            P_qt = torch.nn.Softmax(dim=1)(A_qt/temp_qt) # [K1, K2]
            # Calculate probability that each cluster in 
            # target image is matching to the part mask in
            # the source image.
            part_mask = resize(img_as_bool(part), src_nps).astype(bool)
            P_tq = P_ts_img[:,part_mask]
            P_tq = P_tq.sum(-1) # [K2,]

            # Get scores for each cluster in Target 
            # how likely is it a match to the query
            # And how likely does it match to the query
            S_fg = P_qt.sum(0) * P_tq
            I1 = map_values_to_segments(self.K_target_mapped, S_fg)

            #map_p_tq = map_values_to_segments(C, P_tq)
            #map_p_qt = map_values_to_segments(C, P_qt.sum(0))

            # Set as Unary to CRF
            # Lower energy = bigger distance
            # Higher probability = lower distance.
            P = torch.stack([I1,I2],dim=-1).numpy()
            P = P.reshape(*tgt_nps,-1)
            final = CRF(self.tgt_img_resized, P, 
                        tgt_nps, None, self.tgt_load_size )
            final = final.reshape(*load_shape)

            segm_out[i] = final.reshape(*load_shape)>0
        return segm_out

    def find_correspondences(self, temp_ts=0.02, temp_qt=0.2):
        """ Find correspondences - does both part and point correpsondence
        : param temp_ts: float, Target to support temperature
        : param temp_qt: float, Query to target temperature
        """
        self.segm_out, self.aff_out = None,None
        self.segm_out = self.find_part_correspondences(temp_ts=temp_ts, temp_qt=temp_qt)
        if self.src_query_kps is not None:
            self.aff_out = self.get_kp_correspondence()
        return self.segm_out, self.aff_out

    def get_kp_correspondence(self):
        """ Find point correpsondence """
        print ("Starting processing of the affordances")
        aff_out = []
        sims = self.d_fn(self.src_corr_descs[0,0].to(self.device),
                        self.tgt_corr_descs[0,0].to(self.device)).cpu()
        sims = sims.reshape(self.src_corr_num_patches[0],
                            self.src_corr_num_patches[1],
                            self.tgt_corr_num_patches[0],
                            self.tgt_corr_num_patches[1])

        if self.src_query_kps is None: 
          return None
        for i, kp_i in enumerate(self.src_query_kps):
            if kp_i is None:
                aff_out.append(None)
                continue
            elif len(kp_i)==0:
                aff_out.append(None)
                continue
            else:
                kps = np.asarray(kp_i)
                u_im_k_A,v_im_k_A = rescale_pts(kps[:,0], kps[:,1],
                                    np.asarray(self.src_img).shape,
                                    self.src_load_shape)


                patch, stride = self.model.p, self.model.stride
                u_d_k_A,v_d_k_A = uv_im_to_desc(u_im_k_A, v_im_k_A, 
                                                patch, stride)

                tgt_np = self.tgt_corr_num_patches
                seg_i_rescaled = img_as_bool(resize(self.segm_out[i], tgt_np))>0
                unroll = np.asarray(np.where(seg_i_rescaled==True)) # [2,M]
                if unroll.shape[-1] == 0:
                    aff_out.append(None)
                    continue

                # unroll is xy format
                # Note V corresponds to row and U to column
                sims_aff = sims [to_int(v_d_k_A), 
                                to_int(u_d_k_A)] # [N, H2, W2]
                sims_aff = sims_aff[:,unroll[0],unroll[1]]  # [N, M] 
                val, loc = sims_aff.flatten(1).max(dim=-1)
                xy = torch.Tensor(unroll[:,loc])

                # Swap XY to UV
                u_d_k_B, v_d_k_B = xy[1,:], xy[0,:]
                u_im_k_B,v_im_k_B = uv_desc_to_im(u_d_k_B,v_d_k_B, 
                                                  patch, stride)
                aff_out.append( (u_im_k_B,v_im_k_B) )
        return aff_out
    
    def visualize_pca(self):
        print(self.src_descs.shape, self.tgt_descs.shape)
        feats = np.concatenate([self.src_descs[0,0]/np.linalg.norm(self.src_descs[0,0],axis=-1, keepdims=True),self.tgt_descs[0,0]/np.linalg.norm(self.tgt_descs[0,0],axis=-1, keepdims=True)], axis=0)
        print(feats.shape)

        pca = PCA(n_components=3)
        feats = pca.fit_transform(feats)
        # isomap = Isomap(n_components=3)
        # feats = isomap.fit_transform(feats)
        feats = (feats - feats.min(axis=0)) / (feats.max(axis=0) - feats.min(axis=0))
        fig, axs = plt.subplots(1, 2)
        # first subplot
        axs[0].imshow(feats[:len(self.src_descs[0,0])].reshape((*self.src_num_patches, 3)), interpolation='nearest')
        # second subplot
        axs[1].imshow(feats[len(self.src_descs[0,0]):].reshape((*self.tgt_num_patches, 3)), interpolation='nearest')
        plt.suptitle("PCA of source and target features")
        # plt.suptitle("Isomap of source and target features")
        plt.show()
        exit()


def build_affcorrs(version=1, **kwargs):
    """ Get AffCorrs model based on version """
    if version==1:
        return AffCorrs_V1(kwargs)
    else:
        raise NotImplementedError("Requested version is not implemented yet")