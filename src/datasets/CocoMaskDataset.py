""" Unused. Dataset containing masks from the Coco dataset. """

import torch
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection


class CocoMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root_file="../data/Coco/val2017", ann_file="../data/Coco/annotations/instances_val2017.json", transform=None):
        super().__init__()

        self.full_dataset = CocoDetection(root=root_file, annFile=ann_file)
        self.coco = self.full_dataset.coco

        animal_categories = [
            "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe"
        ]

        animal_cat_ids = self.coco.getCatIds(catNms=animal_categories)


        animal_img_ids = []
        for cat in animal_cat_ids:
            animal_img_ids = animal_img_ids + self.coco.getImgIds(catIds=[cat])

        single_animal_img_ids = []
        for img_id in animal_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)

            cats = {ann["category_id"] for ann in anns if ann["category_id"] in animal_cat_ids}

            if len(cats) == 1:
                single_animal_img_ids.append(img_id)


        img_id_to_index = {img_id: img_idx for img_idx, img_id in enumerate(self.full_dataset.ids)}

        subset_indices = [img_id_to_index[img_id] for img_id in single_animal_img_ids]

        self.dataset = Subset(self.full_dataset, subset_indices)

        self.ann_ids_per_class = 0
        self.animal_ann_per_index = {}
        self.betti_number_per_index = torch.zeros(len(self.dataset))
        org_class_per_index = torch.zeros(len(self.dataset))
        for idx in range(len(self.dataset)):
            img, anns = self.dataset[idx]

            self.animal_ann_per_index[idx] = []
            for ann in anns:
                if ann["category_id"] in animal_cat_ids:
                    self.animal_ann_per_index[idx].append(ann["id"])
                    org_class_per_index[idx] = ann["category_id"]
            self.betti_number_per_index[idx] = len(self.animal_ann_per_index[idx])

        unique_classes, self.class_per_index = torch.unique(org_class_per_index, return_inverse=True)

        self.num_classes = len(unique_classes)
        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        anns = self.coco.loadAnns(self.animal_ann_per_index[idx])

        mask = torch.zeros((img.height, img.width), dtype=torch.uint8)

        for ann in anns:
            mask_one_obj = self.coco.annToMask(ann)
            mask = torch.maximum(mask, torch.tensor(mask_one_obj, dtype=torch.uint8))

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        loss_function = torch.nn.MSELoss()
        loss = loss_function(mask, torch.rand_like(mask))

        return mask, self.class_per_index[idx], self.betti_number_per_index[idx], idx