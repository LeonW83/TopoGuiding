import os.path

from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.datasets.ShapesDataset import ShapesDataset
from src.datasets.CocoMaskDataset import CocoMaskDataset

class TrainingArgs:
    """ Dataset Arguments used for training """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_dataset(name, data_folder="./data", image_size=256):
    """
    Load a dataset given the dataset name.
    :param name: Name to the dataset. Usually "shapes"
    :param data_folder: Path to the folder where the dataset is stored.
    :param image_size: Image size. If image size does not match the real size of the image, the image is scaled to the desired size.
    :return:
    """
    dataset=None
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),  #
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(
            root=data_folder,
            train=True,
            download=True,
            transform=transform
        )
        dataset.image_size = 28
        dataset.channels = 1

    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5,0.5), (0.5,0.5,0.5))
        ])
        dataset = datasets.CIFAR10(
            root=data_folder,
            train=True,
            download=True,
            transform=transform
        )
        dataset.image_size = 32
        dataset.channels = 3
        dataset.num_classes = 10

    elif name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5,0.5), (0.5,0.5,0.5))
        ])
        dataset = datasets.CIFAR100(
            root=data_folder,
            train=True,
            download=True,
            transform=transform
        )
        dataset.image_size = 32
        dataset.channels = 3
        dataset.num_classes = 100

    elif name == "flowers":
        image_size=64
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5,0.5), (0.5,0.5,0.5))
        ])
        dataset = datasets.Flowers102(
            root=data_folder,
            download=True,
            split="test",
            transform=transform
        )
        dataset.image_size = image_size
        dataset.channels = 3
        dataset.num_classes = 102

    elif name == "coco":

        def transform_coco(img, mask, img_size=256):

            mask = mask.unsqueeze(0)

            img = TF.resize(img, [img_size, img_size])
            mask = TF.resize(mask, [img_size, img_size], interpolation=InterpolationMode.NEAREST)

            img = TF.to_tensor(img)

            # mask to diffusion/flow matching format
            mask = mask.float()
            mask = mask * 2 - 1

            img = img * 2 - 1

            return img, mask

        ann_file_path = os.path.join(data_folder, "coco/annotations/instances_val2017.json")
        root_folder_path = os.path.join(data_folder, "coco/val2017")

        dataset = CocoMaskDataset(root_file=root_folder_path, ann_file=ann_file_path, transform=lambda img, mask: transform_coco(img, mask, img_size=image_size))
        dataset.image_size = 64
        dataset.channels = 1

    elif name == "shapes":
        def transform_shapes(mask, img_size=256):

            mask = mask.unsqueeze(0)
            mask = TF.resize(mask, [img_size, img_size], interpolation=InterpolationMode.NEAREST)

            # mask to diffusion/flow matching format
            mask = mask.float()
            mask = mask * 2 - 1

            return mask

        root_path = os.path.join(data_folder, "shapes_dataset")
        dataset = ShapesDataset(root_path=root_path, transform=lambda mask: transform_shapes(mask, img_size=image_size))
        dataset.image_size = image_size
        dataset.channels = 1
        dataset.num_classes = 3

    data_args = TrainingArgs(image_size=dataset.image_size, num_channels=dataset.channels, num_classes=dataset.num_classes)


    return dataset, data_args


