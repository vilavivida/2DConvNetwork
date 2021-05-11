import skimage.data
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
import numpy as np
import cv2

class kernel(object):
    def __init__(self, array2D):
        self.l1_filter = np.array(array2D) 

class convLayer(object):
    def __init__(self, img, kernel):
        self.img = img
        self.kernel = np.flipud(np.fliplr(kernel.l1_filter))

        self.xKernShape = self.kernel.shape[0]
        self.yKernShape = self.kernel.shape[1]
        self.xImgShape = self.img.shape[0]
        self.yImgShape = self.img.shape[1]

        self.check_kernel()

    def check_kernel(self):
        # check if number of image channels matches the kernel depth
        if len(self.img.shape) > 2 or len(self.kernel.shape) > 3:
            assert image.shape[-1] == l1_filter[-1], "Error: number of channels in both image and kernel must match"
        # check if kernel is squared
        assert self.kernel.shape[0] == self.kernel.shape[1], "Error: kernel should be square matrix"
        # check if kernel dimensions are odd
        assert self.kernel.shape[1] % 2 != 0, "Error: number of rows and columns must be odd"

    def conv2D(self, padding = 0, stride = 1):
        xOutput = int(((self.xImgShape - self.xKernShape + 2 * padding) / stride) + 1)
        yOutput = int(((self.xImgShape - self.yKernShape + 2 * padding) / stride) + 1) # squared

        # initialize
        feature_map = np.zeros((xOutput, yOutput))

        # add padding
        if padding != 0:
            # adding layers of zeros to input image
            imgPadded = np.zeros((self.xImgShape + padding * 2, self.yImgShape + padding * 2))
            imgPadded[int(padding) : int(-1 * padding), int(padding) : int(-1 * padding)] = self.img
        else:
            imgPadded = self.img

        # apply kernel
        for y in range(self.yImgShape):
            if y > self.yImgShape - self.yKernShape:
                break
            if y % stride == 0:
                for x in range(self.xImgShape):
                    # Go to next row once kernel is out of bounds
                    if x > self.xImgShape - self.xKernShape:
                        break
                    try:
                        if x % stride == 0:
                            feature_map[x, y] = (self.kernel * imgPadded[x: x + self.xKernShape, y: y + self.yKernShape]).sum()
                    except:
                        break

        return feature_map

class reluLayer(object):
    """
    serves to break up the linearity in the image
    """
    def __init__(self):
        pass

    def rectify(self, feature_map):
        """
        Return the original value in the feature map if it is larger than 0
        Otherwise, return 0
        """
        output = np.maximum(0, feature_map)
        return output

class poolingLayer(object):
    """
    The filter selects the maximum or calculates the mean out of the pixels covered under the kernel
    """
    def __init__(self, kernel):
        self.xKernShape = kernel.l1_filter.shape[0]
        self.yKernShape = kernel.l1_filter.shape[1]

    def pool2d(self, feature_map, stride = 1, poolMode = "max"):
        output_shape = ((feature_map.shape[0] - self.xKernShape)//stride + 1,
                        (feature_map.shape[1] - self.yKernShape)//stride + 1)

        kernel_size = (self.xKernShape, self.yKernShape)

        # use numpy function to create a view into the array with the given shape and stride
        feature_map = as_strided(feature_map, 
                                 shape = output_shape + kernel_size, 
                                 strides = (stride * feature_map.strides[0], stride * feature_map.strides[1]) + feature_map.strides)
        feature_map  = feature_map.reshape(-1, *kernel_size)

        if poolMode == "max":
            return feature_map.max(axis=(1,2)).reshape(output_shape)
        elif poolMode == "mean":
            return feature_map.mean(axis=(1,2)).reshape(output_shape)


def compile_model(img, array1, array2, array3):
    kernel_1 = kernel(array1)

    print("\n ---------- Working with the First Conv Layer ----------")
    l1_Conv = convLayer(img, kernel_1).conv2D() # input the original image
    print("\n ---------- ReLU ----------")
    l1_Conv_Relu = reluLayer().rectify(l1_Conv)
    print("\n ---------- Pooling ----------")
    l1_Conv_Relu_Pool = poolingLayer(kernel_1).pool2d(l1_Conv_Relu)
    print("\n ---------- End of the First Conv Layer ----------")

    kernel_2 = kernel(array2)
    print("\n ---------- Working with the Second Conv Layer ----------")
    l2_Conv = convLayer(l1_Conv_Relu_Pool, kernel_2).conv2D() # input the output from the first round
    print("\n ---------- ReLU ----------")
    l2_Conv_Relu = reluLayer().rectify(l2_Conv)
    print("\n ---------- Pooling ----------")
    l2_Conv_Relu_Pool = poolingLayer(kernel_2).pool2d(l2_Conv_Relu)
    print("\n ---------- End of the Second Conv Layer ----------")

    kernel_3 = kernel(array3)
    print("\n ---------- Working with the Third Conv Layer ----------")
    l3_Conv = convLayer(l2_Conv_Relu_Pool, kernel_3).conv2D() # input the output from the second round
    print("\n ---------- ReLU ----------")
    l3_Conv_Relu = reluLayer().rectify(l3_Conv)
    print("\n ---------- Pooling ----------")
    l3_Conv_Relu_Pool = poolingLayer(kernel_3).pool2d(l3_Conv_Relu)
    print("\n ---------- End of the Third Conv Layer ----------")

    # plot
    f, axarr = plt.subplots(3,3, figsize=(10, 12))

    # l1
    axarr[0,0].imshow(l1_Conv)
    axarr[0,0].set_title('l1_Conv')

    axarr[0,1].imshow(l1_Conv_Relu)
    axarr[0,1].set_title('l1_Conv_Relu')

    axarr[0,2].imshow(l1_Conv_Relu_Pool)
    axarr[0,2].set_title('l1_Conv_Relu_Pool')

    # l2
    axarr[1,0].imshow(l2_Conv)
    axarr[1,0].set_title('l2_Conv')

    axarr[1,1].imshow(l2_Conv_Relu)
    axarr[1,1].set_title('l2_Conv_Relu')

    axarr[1,2].imshow(l2_Conv_Relu_Pool)
    axarr[1,2].set_title('l2_Conv_Relu_Pool')

    # l3
    axarr[2,0].imshow(l3_Conv)
    axarr[2,0].set_title('l3_Conv')

    axarr[2,1].imshow(l3_Conv_Relu)
    axarr[2,1].set_title('l3_Conv_Relu')

    axarr[2,2].imshow(l3_Conv_Relu_Pool)
    axarr[2,2].set_title('l3_Conv_Relu_Pool')

    f.savefig('output.png', bbox_inches='tight')
    plt.show()

# pull a random image from skimage.data
img = skimage.data.astronaut()
img = skimage.color.rgb2gray(img)

# define filters
array1 = np.random.randint(3, size = (3, 3))
array2 = np.random.randint(4, size = (3, 3))
array3 = np.random.randint(5, size = (3, 3))

compile_model(img, array1, array2, array3)
