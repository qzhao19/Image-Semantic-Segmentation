from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def write_images(image):
    """This function focus on coloring tag information in ground truth image. And ploy colored image
    
    """
    Unlabeled    = [0, 0, 0]
    Building     = [70, 70, 70]
    Fence        = [190, 153, 153]
    Other        = [250, 170, 160]
    Pedestrian   = [220, 20, 60]
    Pole         = [153, 153, 153]
    Road_Line    = [157, 234, 50]
    Road         = [128, 64, 128]
    Sidewalk     = [244, 35, 232]
    Vegetation   = [107, 142, 35]
    Car          = [0, 0, 142]
    Wall         = [102, 102, 156]
    Traffic_Sign = [220, 220, 0]
    
    outputs = np.zeros((image.shape[0], image.shape[1], 3))
    
    tags=np.array([Unlabeled, Building, 
                   Fence, Other, 
                   Pedestrian, Pole, 
                   Road_Line, Road, 
                   Sidewalk, Vegetation, 
                   Car, Wall, Traffic_Sign])
    
    for tag in range(len(tags)):
        outputs[:,:,0] += (image == tag)*(tags[tag][0]).astype('uint8')
        outputs[:,:,1] += (image == tag)*(tags[tag][1]).astype('uint8')
        outputs[:,:,2] += (image == tag)*(tags[tag][2]).astype('uint8')
        
    outputs = Image.fromarray(np.uint8(outputs))
    
    return outputs

def save_images(image, filename):
    """ store label data to colored image """
    Unlabeled    = [0, 0, 0]
    Building     = [70, 70, 70]
    Fence        = [190, 153, 153]
    Other        = [250, 170, 160]
    Pedestrian   = [220, 20, 60]
    Pole         = [153, 153, 153]
    Road_Line    = [157, 234, 50]
    Road         = [128, 64, 128]
    Sidewalk     = [244, 35, 232]
    Vegetation   = [107, 142, 35]
    Car          = [0, 0, 142]
    Wall         = [102, 102, 156]
    Traffic_Sign = [220, 220, 0]
    
    
    r = image.copy()
    g = image.copy()
    b = image.copy()
#     label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    
    label_colours=np.array([Unlabeled, Building, 
                   Fence, Other, 
                   Pedestrian, Pole, 
                   Road_Line, Road, 
                   Sidewalk, Vegetation, 
                   Car, Wall, Traffic_Sign])
    
    
    for l in range(len(label_colours)):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    output_image = Image.fromarray(np.uint8(rgb))
    output_image.save(filename)



def plots_images(images, labels, predicted_labels, uncertainty):
    
    num_images = len(images)
    
    cols = ['Input', 'Ground truth', 'Output', 'Uncertainty']
    rows = ['Image {}'.format(row) for row in range(1,num_images+1)]
    #rows = ['Worst', 'Average', 'Best']

    fig, axes = plt.subplots(nrows=num_images, ncols=4, figsize=(20,num_images*4))

    for i in range(num_images):

        plt.subplot(num_images, 4, (4*i+1))
        plt.imshow(images[i])
        #plt.ylabel("Image %d" % (i+1), size='18')
        plt.ylabel(rows[i], size='22')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[0], size='22', va='bottom')

        plt.subplot(num_images, 4, (4*i+2))
        write_images(labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[1], size='22', va='bottom')

        plt.subplot(num_images, 4, (4*i+3))
        write_images(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[2], size='22', va='bottom')
            
        plt.subplot(num_images, 4, (4*i+4))
        plt.imshow(uncertainty[i], cmap = 'Greys')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[3], size='22', va='bottom')
    plt.show()
    plt.close()
    

    
    
def plots_images_external(images, predicted_labels, uncertainty):
    """
    
    """
    
    num_images = len(images)
    
    cols = ['Input', 'Output', 'Uncertainty']
    rows = ['Image {}'.format(row) for row in range(1,num_images+1)]
    #rows = ['Worst', 'Average', 'Best']

    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(16,num_images*4))
    

    for i in range(num_images):

        plt.subplot(num_images, 3, (3*i+1))
        plt.imshow(images[i])
        #plt.ylabel("Image %d" % (i+1), size='18')
        plt.ylabel(rows[i], size='18')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[0], size='18', va='bottom')

        plt.subplot(num_images, 3, (3*i+2))
        write_images(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[1], size='18', va='bottom')
            
        plt.subplot(num_images, 3, (3*i+3))
        plt.imshow(uncertainty[i], cmap = 'Greys')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[2], size='18', va='bottom')

    plt.show()
    plt.close()