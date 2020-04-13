import numpy as np
import cv2
import os
def preprocess(image):
    # crop the original image (300,300,3) to (256,256,3)
    return image[22:278,22:278]

def load_dataset(dataset_dir):
    classes = ['horses', 'humans']
    dataset = []

    for i, c in enumerate(classes):
        path = os.path.join(dataset_dir, c)
        path_files = os.listdir(path)
        for pf in path_files:
            image = cv2.imread(os.path.join(path,pf))
            image = preprocess(image)
            dataset.append((image,i))
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    data, labels = dataset[:,0], dataset[:,1]
    # data = data.reshape(-1,1)
    labels = labels.reshape(-1,1)
    return data, labels

if __name__ == '__main__':
    data, labels = load_dataset('./human_horse_dataset/train')
    print(data.shape, labels.shape)
