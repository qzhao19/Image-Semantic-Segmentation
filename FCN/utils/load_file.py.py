import os

def load_image_label_file(file_path):
    """The function is to read image
    Args:
        file_path: a string, containing image file path
    Return:
        image filename list and label filename list
    """
    image_filename_list = []
    label_filename_list = []
    if not os.path.exists(file_path):
        raise ValueError('file could not be opened, perhaps you need to check it')
    # open file
    with open(file_path, 'r') as file:
        lines = [line.strip().split(' ') for line in file.readlines()]
    for i in range(len(lines)):
        image_filename_list.append(lines[i][0])
        label_filename_list.append(lines[i][0])
    
    return image_filename_list, label_filename_list
