import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from detectors.utils import projection
from detectors.datasets.WildTrack import Wildtrack

if __name__ == '__main__':
    img = Image.open('/root/deep_learning/dzc/data/Wildtrack_dataset/Image_subsets/C1/00000000.png')
    dataset = Wildtrack('/root/deep_learning/dzc/data/Wildtrack_dataset')
    xi = np.arange(0, 480, 40)
    yi = np.arange(0, 1440, 40)
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_coord = dataset.get_worldcoord_from_worldgrid(world_grid)

    mid = np.array([[0, 240], [1439, 720], [0, 20]]).reshape([3, -1])
    
    world_coord = dataset.get_worldcoord_from_worldgrid_3d(mid)

    # world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([3, -1])
    # img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[0],
    #                                                       dataset.extrinsic_matrices[0])
    # print(world_coord[0][11], world_coord[1][11])
    img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[0],
                                                       












   dataset.extrinsic_matrices[0])
    print(img_coord.T)
    img_coord = img_coord[:, np.where((img_coord[0] > 0) & (img_coord[1] > 0) &
                                      (img_coord[0] < 1920) & (img_coord[1] < 1080))[0]]
    print(img_coord.T)


    plt.imshow(img)
    plt.show()
    img_coord = img_coord.astype(int).transpose()
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # print("asdasd", img_coord)
    for point in img_coord:
        # print("dzc", point)
        cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img.save('img_grid_visualize.png')
    plt.imshow(img)
    plt.show()
    pass