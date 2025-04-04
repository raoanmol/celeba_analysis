import os

def load_data_annotations(file_path = None, attribute = 'Male'):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '../data/list_attr_celeba.txt')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    attribute_names = lines[1].strip().split()

    if attribute not in attribute_names:
        raise ValueError(f"Attribute '{attribute}' not found in the attribute list.")
    
    attribute_index = attribute_names.index(attribute)

    annotations = []

    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) != len(attribute_names) + 1:
            continue

        annotations.append(int(parts[attribute_index + 1]))

    train_end = 162770
    val_end = 182637

    train_annotations = annotations[:train_end]
    validation_annotations = annotations[train_end:val_end]
    test_annotations = annotations[val_end:]

    return train_annotations, validation_annotations, test_annotations
