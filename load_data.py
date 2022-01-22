import numpy as np
import os
import random
import torch

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [[i, os.path.join(path, image)]
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return np.array(images_labels, dtype=object)


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image

def load_imgs_from_folders(folders, dim_input):
    all_characters = []
    for folder in folders:
        character = []
        for img_pth in os.listdir(folder):
            img = image_file_to_array(os.path.join(folder, img_pth), dim_input)
            character.append(img)
        all_characters.append(character)
    return np.array(all_characters)

class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}, device = torch.device('cpu'), in_mem=True):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)
            
            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.
                
            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class  # K (number of shots)
        self.num_classes = num_classes  # N (number of ways)
        self.in_mem = in_mem

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device
        
        # Load all images to speed up training
        if self.in_mem:
            self.metatrain_characters = load_imgs_from_folders(self.metatrain_character_folders, self.dim_input)
            self.metaval_characters = load_imgs_from_folders(self.metaval_character_folders, self.dim_input)
            self.metatest_characters = load_imgs_from_folders(self.metatest_character_folders, self.dim_input)
        
    
    
    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from
                
            batch_size: int:
                Size of batch of tasks to sample
                
        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes
                
            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if self.in_mem:
            return self.sample_batch_mem(batch_type, batch_size)
        
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        all_samples = []
        all_labels = []
        # Sample B batches
        for _ in range(batch_size):
            # Sample N classes
            N_classes = random.sample(folders, self.num_classes)
            one_hots = np.eye(self.num_classes).tolist()

            # Sample K+1 for each class
            samples = get_images(N_classes, one_hots, nb_samples=self.num_samples_per_class+1, shuffle=False)
            img_paths = samples[:,1]

            imgs = []
            for img_pth in img_paths:
                imgs.append(image_file_to_array(img_pth, self.dim_input))
            imgs = np.array(imgs).reshape(self.num_classes,self.num_samples_per_class+1,self.dim_input)
            all_samples.append(imgs)
            all_labels.append(np.array(samples[:,0].tolist()).reshape(self.num_classes,self.num_samples_per_class+1,self.num_classes))
        
        all_samples = torch.tensor(all_samples).float().to(self.device).permute(0,2,1,3)
        all_labels = torch.tensor(all_labels).long().to(self.device).permute(0,2,1,3)

        return all_samples, all_labels
    
    
    def sample_batch_mem(self, batch_type, batch_size):
        if batch_type == "train":
            characters = self.metatrain_characters
        elif batch_type == "val":
            characters = self.metaval_characters
        else:
            characters = self.metatest_characters

        all_samples = []
        all_labels = []
        # Sample B batches
        for _ in range(batch_size):
            class_indices = np.random.randint(0,characters.shape[0], self.num_classes)
            N_classes = characters[class_indices]
            
            # Sample K+1 for each class
            samples = []
            labels = []
            for idx,c in enumerate(N_classes):
                # Reminder: need to shuffle
                indices = np.random.randint(0,c.shape[0],self.num_samples_per_class+1)
                samples.append(c[indices])
                lbl = np.zeros((self.num_samples_per_class+1, self.num_classes))
                lbl[:, idx] = 1
                labels.append(lbl.tolist())
            samples = np.array(samples)
            all_samples.append(samples.tolist())
            all_labels.append(labels)
        
        all_samples = torch.tensor(all_samples).float().to(self.device).permute(0,2,1,3)
        all_labels = torch.tensor(all_labels).long().to(self.device).permute(0,2,1,3)

        return all_samples, all_labels