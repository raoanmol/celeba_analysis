import os

def load_partitioned_data(file_path = None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '../data/list_eval_partition.txt')
    
    train_images = []
    validation_images = []
    test_images = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name, partition = parts[0].strip(), parts[1]
                if partition == '0':
                    train_images.append(os.path.join(os.path.abspath('./data/img_align_celeba/'), image_name))
                elif partition == '1':
                    validation_images.append(os.path.join(os.path.abspath('./data/img_align_celeba/'), image_name))
                elif partition == '2':
                    test_images.append(os.path.join(os.path.abspath('./data/img_align_celeba/'), image_name))

    return train_images, validation_images, test_images