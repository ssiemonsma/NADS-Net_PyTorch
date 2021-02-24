import h5py
import json
import numpy as np
import cv2
from py_rmpe_transformer import Transformer, AugmentSelection
from py_rmpe_heatmapper import Heatmapper
from torchvision import transforms


class Dataset_Generator():
    def __init__(self, global_config, config, include_background_output, using_Aisin_output_format, augment=True):
        self.global_config = global_config
        self.config = config
        self.include_background_output = include_background_output
        self.using_Aisin_output_format = using_Aisin_output_format
        self.augment = augment

        self.h5_filename = config.source()
        self.h5 = h5py.File(self.h5_filename, "r")

        self.dataset = self.h5['dataset']
        self.images = self.h5['images']
        self.masks = self.h5['masks']

        self.heatmapper = Heatmapper(global_config)
        self.transformer = Transformer(global_config)

        self.keys = list(self.dataset.keys())

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.Aisin_keypoint_indices = [4, 3, 2, 1, 5, 6, 7, 8, 11]
        self.Aisin_PAF_indices = [12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 0, 1, 6, 7]

    def read_data(self, key):
        entry = self.dataset[key]

        assert 'meta' in entry.attrs, "No 'meta' attribute in .h5 file. Did you generate .h5 with new code?"

        meta = json.loads(entry[()])
        debug = json.loads(entry.attrs['meta'])
        meta = self.config.convert(meta, self.global_config)

        image = self.images[meta['image']][()]
        mask_miss = None

        if len(image.shape)==2 and image.shape[1]==1:
            image = cv2.imdecode(image, flags=-1)

        if image.shape[2]>3:
            mask_miss = image[:, :, 3]
            image = image[:, :, 0:3]

        if mask_miss is None:
            if self.masks is not None:
                mask_miss = self.masks[meta['image']][()]
                if len(mask_miss.shape) == 2 and mask_miss.shape[1]==1:
                    mask_miss = cv2.imdecode(mask_miss, flags = -1)

        if mask_miss is None:
            mask_miss = np.full((image.shape[0], image.shape[1]), fill_value=255, dtype=np.uint8)

        return image, mask_miss, meta, debug

    def convert_to_Aisin_format(self, keypoint_heatmap_labels, PAF_labels, keypoint_heatmap_masks, PAF_masks):
        keypoint_heatmap_labels = keypoint_heatmap_labels[:,:,self.Aisin_keypoint_indices]
        keypoint_heatmap_masks = keypoint_heatmap_masks[:,:,self.Aisin_keypoint_indices]

        if self.include_background_output:
            background = 1 - np.amax(keypoint_heatmap_labels, axis=2)
            keypoint_heatmap_labels = np.concatenate([keypoint_heatmap_labels, background[..., np.newaxis]], axis=2)

            background_mask = np.ones((96, 96, 1))
            keypoint_heatmap_masks = np.concatenate([keypoint_heatmap_masks, background_mask], axis=2)

        PAF_labels = PAF_labels[:,:,self.Aisin_PAF_indices]
        PAF_masks = PAF_masks[:,:,self.Aisin_PAF_indices]

        return keypoint_heatmap_labels, PAF_labels, keypoint_heatmap_masks, PAF_masks

    def __getitem__(self, index):
        image, mask, meta, debug = self.read_data(self.keys[index])

        # Data augmentation
        assert mask.dtype == np.uint8
        image, mask, meta = self.transformer.transform(image, mask, meta, aug=None if self.augment else AugmentSelection.unrandom())
        assert mask.dtype == np.float

        # We need a layered mask on next stage
        mask = self.config.convert_mask(mask, self.global_config, joints=meta['joints'])

        # Create keypoint heatmaps and PAFs
        labels = self.heatmapper.create_heatmaps(meta['joints'], mask)

        keypoint_heatmap_masks = mask[:, :, self.global_config.paf_layers:]
        PAF_masks = mask[:, :, :self.global_config.paf_layers]

        keypoint_heatmap_labels = labels[:, :, self.global_config.paf_layers:]
        PAF_labels = labels[:, :, :self.global_config.paf_layers]

        if self.using_Aisin_output_format:
            # Eliminate keypoints and PAF layers not present in Aisin dataset
            keypoint_heatmap_labels, PAF_labels, keypoint_heatmap_masks, PAF_masks = self.convert_to_Aisin_format(keypoint_heatmap_labels, PAF_labels, keypoint_heatmap_masks, PAF_masks)

        # Move the channel dimension to the correct PyTorch position
        keypoint_heatmap_masks, PAF_masks, keypoint_heatmap_labels, PAF_labels = list(map(lambda x: np.rollaxis(x, 2), [keypoint_heatmap_masks, PAF_masks, keypoint_heatmap_labels, PAF_labels]))

        image = image.astype(np.float32)
        image = self.preprocess(np.flip(image, 2)/255)
        # image = np.rollaxis(image, 2)

        return image, keypoint_heatmap_masks, PAF_masks, keypoint_heatmap_labels, PAF_labels

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        if 'h5' in vars(self):
            self.h5.close()


class CanonicalConfig:
    def __init__(self):
        self.width = 384
        self.height = 384

        self.stride = 4

        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.parts += ["background"]
        self.num_parts_with_background = len(self.parts)

        leftParts, rightParts = CanonicalConfig.ltr_parts(self.parts_dict)
        self.leftParts = leftParts
        self.rightParts = rightParts

        # this numbers probably copied from matlab they are 1.. based not 0.. based
        self.limb_from =  ['neck', 'Rhip', 'Rkne', 'neck', 'Lhip', 'Lkne', 'neck', 'Rsho', 'Relb', 'Rsho', 'neck', 'Lsho', 'Lelb', 'Lsho',
         'neck', 'nose', 'nose', 'Reye', 'Leye']
        self.limb_to = ['Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Rsho', 'Relb', 'Rwri', 'Rear', 'Lsho', 'Lelb', 'Lwri', 'Lear',
         'nose', 'Reye', 'Leye', 'Rear', 'Lear']

        self.limb_from = [ self.parts_dict[n] for n in self.limb_from ]
        self.limb_to = [ self.parts_dict[n] for n in self.limb_to ]

        assert self.limb_from == [x-1 for x in [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]]
        assert self.limb_to == [x-1 for x in [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]]

        self.limbs_conn = list(zip(self.limb_from, self.limb_to))

        self.paf_layers = 2*len(self.limbs_conn)
        self.heat_layers = self.num_parts
        self.num_layers = self.paf_layers + self.heat_layers + 1

        self.paf_start = 0
        self.heat_start = self.paf_layers
        self.bkg_start = self.paf_layers + self.heat_layers

        #self.data_shape = (self.height, self.width, 3)     # 368, 368, 3
        self.mask_shape = (self.height//self.stride, self.width//self.stride)  # 46, 46
        self.parts_shape = (self.height//self.stride, self.width//self.stride, self.num_layers)  # 46, 46, 57

        class TransformationParams:
            def __init__(self):
                self.target_dist = 0.6;
                self.scale_prob = 1;  # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
                self.scale_min = 0.5;
                self.scale_max = 1.1;
                self.max_rotate_degree = 40.
                self.center_perterb_max = 40.
                self.flip_prob = 0.5
                self.sigma = 7.
                self.paf_thre = 8.  # it is original 1.0 * stride in this program

        self.transform_params = TransformationParams()

    @staticmethod
    def ltr_parts(parts_dict):
        # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
        leftParts  = [ parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"] ]
        rightParts = [ parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"] ]
        return leftParts, rightParts


class COCOSourceConfig:
    def __init__(self, hdf5_source):
        self.hdf5_source = hdf5_source
        self.parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
             'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']

        self.num_parts = len(self.parts)

        # for COCO neck is calculated like mean of 2 shoulders.
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))

    def convert(self, meta, global_config):
        joints = np.array(meta['joints'])

        assert joints.shape[1] == len(self.parts)

        result = np.zeros((joints.shape[0], global_config.num_parts, 3), dtype=np.float)
        result[:,:,2]=3.  # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible

        for p in self.parts:
            coco_id = self.parts_dict[p]

            if p in global_config.parts_dict:
                global_id = global_config.parts_dict[p]
                assert global_id!=1, "neck shouldn't be known yet"
                result[:,global_id,:]=joints[:,coco_id,:]

        if 'neck' in global_config.parts_dict:
            neckG = global_config.parts_dict['neck']
            RshoC = self.parts_dict['Rsho']
            LshoC = self.parts_dict['Lsho']

            # no neck in coco database, we calculate it as average of shoulders
            # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
            both_shoulders_known = (joints[:, LshoC, 2]<2)  &  (joints[:, RshoC, 2] < 2)

            result[~both_shoulders_known, neckG, 2] = 2. # otherwise they will be 3. aka 'never marked in this dataset'

            result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                        joints[both_shoulders_known, LshoC, 0:2]) / 2
            result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                                     joints[both_shoulders_known, LshoC, 2])

        meta['joints'] = result

        return meta

    def convert_mask(self, mask, global_config, joints = None):
        mask = np.repeat(mask[:,:,np.newaxis], global_config.num_layers, axis=2)
        return mask

    def source(self):
        return self.hdf5_source