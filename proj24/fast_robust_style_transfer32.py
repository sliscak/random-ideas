""""
    Fast and Robust Image StyleTransfer and Colorization by providing INPUT and OUTPUT example pairs, saving patches into index, and using similarity search over the index with the incoming input image patches.
    Training: break down both INPUT and OUTPUT example images into patches/tiles a associate/link them.
    Inference: break down user input image into patches/tiles, for every patch/tile: similarity search the index, finding most similar patch/tile.
        Get the index of the most similar patch/tile, save a associated colored patch/tile from the dictionary or list or other index into new tensor/array. (all patches/tiles are saved into the tensor/array)
        Use the 'torch.nn.functional.fold' function to stitch/combine all patches/tiles of the tensor/array together into a new image.
        Return the new image.
        Or if the stride value (for the fold function) is equal to kernel size we could replace the patches/tiles directly on the image and so make new image.
        (replacing the patches on the image with the new colored patches/tiles)

    If doing inference: If the distance between the query patch and the patch saved in the index is less than the threshold value,
        use the query patch as input for the fold function. (the patch stays the same, is not replaced or modified)(That is that the region of the image where the patch is stays the same, no change happens, but only if the stride is greater that the size of the patch/kernel)

    DONE: remove/skip duplicate patterns/kernels from faiss index/memory
    TODO: learn/train at lower resolution
    TODO: rotate and mirror the patterns/kernels and use other transformations and augmentations.
    TODO: increase speed by parallelizing the pattern retrieval(similarity search)
    TODO: add a small cache for recently found(retrieved) patterns.
    TODO: turn input patches grayscale and save such patches into the index,
        the patches associated to the grayscale patches stay colored.
        (grayscale patches will be linked to colored patches).
    TODO: save a hierarchy of patches of different resolution?
    TODO: associate/link patches/tiles spatialy(in space) with the help of a graph (or a python Dictionary)
    TODO: fix bug where sometimes after index-pretraining the search faills
    TODO: normalize distance values??
    TODO: use grayscale patches as input saved into index and query

    USAGE-> enter command: "streamlit run fast_robust_style_transfer31.py"
"""

import os
import faiss
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx
from time import time
from collections import Counter, namedtuple
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
from pyvis.network import Network # TODO: rename??
import streamlit.components.v1 as components

def preprocess(bytes_image, image_size=(64, 64), gray_scale=False):
    """"
        Resizes the image and returns it as numpy array.
    """
    image = Image.open(bytes_image)
    image = image.resize(image_size)
    if gray_scale:
        image = ImageOps.grayscale(image)
        # normalize
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=2)
    else:
        # normalize
        image = np.array(image) / 255
        if len(image.shape) < 3:
            return None
    return image


def load_img(path: str = "image.png", image_size=(64, 64), gray_scale=False, return_pair=False):
    """"
        Loads an image, resizes it and returns it as numpy array.
        Options include: output image as grayscale and output grayscale and colored image pair
    """
    image = Image.open(path)
    image = image.resize(image_size)
    if return_pair:
        gray_image = ImageOps.grayscale(image)
        gray_image = np.array(gray_image) / 255
        gray_image = np.expand_dims(gray_image, axis=2)

        image = np.array(image) / 255
        if len(image.shape) < 3:
            return None
        else:
            return (gray_image, image)
    else:
        if gray_scale:
            image = ImageOps.grayscale(image)
            # normalize
            image = np.array(image) / 255
            image = np.expand_dims(image, axis=2)
        else:
            # normalize
            image = np.array(image) / 255
            if len(image.shape) < 3:
                return None
        return image


class ImageDataset(Dataset):
    def __init__(self, path='', image_size=(64, 64), size=None, testing=False):
        self.image_folder_path = path
        self.data = []

        # TODO: Load on the fly
        for root, dirs, files in os.walk(path, topdown=False):
            for image_name in files:
                if '.JPEG' in image_name:
                    image_path = self.image_folder_path + image_name
                    pair = load_img(image_path, image_size=image_size, return_pair=True)
                    if pair is None:
                        continue
                    else:
                        gray_image, image = pair
                    self.data.append((gray_image, image))
                if size is not None:
                    if len(self.data) >= size:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, gray_image = self.data[idx]
        x = image
        y = gray_image
        sample = x, y
        return sample


PatchConnections = namedtuple('PatchConnections', ['up', 'down', 'left', 'right'])

class NeuralMem(nn.Module):
    def __init__(self, image_size=(64, 64), index_pretrain=False, kernel_size=(32, 32), stride:int=1, padding:int=10, threshold:int=0.5):
        super(NeuralMem, self).__init__()
        # res = faiss.StandardGpuResources()
        # self.mem = faiss.IndexFlatL2(25) # size of one tile/kernel
        # self.mem = faiss.index_cpu_to_gpu(res, 0, self.mem)
        self.threshold = threshold
        self.output_size = image_size
        self.kernel = kernel_size
        self.dimensions = int(np.product(self.kernel) * self.output_size[2])
        self.stride = stride
        self.padding = padding
        self.pattern_mappings = {}
        self.index_pretrain = index_pretrain
        # self.graph = networkx.Graph()
        self.graph = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        self.nlist = 100
        self.patches_meta = []
        # self.mem = faiss.IndexFlatL2(self.dimensions)
        if self.index_pretrain:
            self.quantizer = faiss.IndexFlatL2(self.dimensions)
            self.mem = faiss.IndexIVFFlat(self.quantizer, self.dimensions, self.nlist)

        else:
            # self.quantizer = faiss.IndexFlatL2(self.dimensions)
            # self.mem = faiss.IndexIVFFlat(self.quantizer, self.dimensions, self.nlist)
            self.mem = faiss.IndexFlatL2(self.dimensions)

        self.mem2 = faiss.IndexFlatL2(self.dimensions)
        self.num_horizontal_patches = int((TRAINING_IMAGE_SIZE[0] - KERNEL_SIZE[0] + 2 * PADDING) / STRIDE) + 1
        self.num_vertical_patches = int((TRAINING_IMAGE_SIZE[1] - KERNEL_SIZE[1] + 2 * PADDING) / STRIDE) + 1


    def forward(self, image):
        """"
            Input is a image tensor in shape (channels, height, width) where channels is 3
        """
        output_size = image.shape

        # st.write(image.shape)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        with st.spinner('INFERENCE in progress...'):
            progress_bar = st.progress(0)
            unfolded = unfolded.contiguous().numpy().astype('float32')
            ds, ks = self.mem.search(unfolded, 1) # ds -> distances ;
            out = None
            st.write(f'SHAPE: {unfolded.shape}')
            # all_mappings = [int(mappings_id[0]) for mappings_id in ks]
            # for mapping in all_mappings:
            #
            # st.write(ds)
            for i, mappings_id in enumerate(ks):
                if ds[i] >= self.threshold:
                    found = torch.tensor(unfolded[i]).unsqueeze(0)
                else:
                    # st.write(unfolded[i])
                    # st.write(ds[i])
                    # st.exit()
                    mappings_id = int(mappings_id[0])
                    candidates = self.patches_meta[mappings_id]
                    # breakpoint()
                    # if len(self.pattern_mappings[mappings_id].most_common(10)) > 2:
                    #     st.write(f'MOST COMMON: {self.pattern_mappings[mappings_id].most_common(3)}')
                    pattern_id = self.pattern_mappings[mappings_id].most_common(1)[0][0]
                    reconstructed = self.mem2.reconstruct(pattern_id)
                    found = torch.tensor(reconstructed).unsqueeze(0)

                if out is None:
                    out = found
                else:
                    out = torch.cat((out, found), 0)
                progress_bar.progress(i / (unfolded.shape[0] - 1))
        out = out.permute(1, 0)
        out = out.unsqueeze(0)
        out = torch.nn.functional.fold(out,
                                       output_size=output_size[0:2],
                                       kernel_size=self.kernel,
                                       stride=self.stride,
                                       padding=self.padding)
        out = out.squeeze(0).squeeze(0) / out.flatten().max()
        out = out.permute(1, 2, 0)
        # st.write(f'Out shape: {out.shape}')
        return out

    def add(self, input_example, output_example): # adds/appends patches to the index
        # takes two tensor arrays as input.
        # input shape of each example is HxWxC and is changed into CxHxW
        # both examples need to have the same resolution
        t0 = time()
        image1 = input_example.permute(2, 0, 1)
        image2 = output_example.permute(2, 0, 1)

        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)
        unfolded1 = torch.nn.functional.unfold(image1, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded2 = torch.nn.functional.unfold(image2, kernel_size=self.kernel, stride=self.stride, padding=self.padding)

        unfolded1 = unfolded1.squeeze(0)
        unfolded2 = unfolded2.squeeze(0)

        unfolded1 = unfolded1.permute(1, 0)
        unfolded2 = unfolded2.permute(1, 0)

        # TODO: replace with patch_id_map = np.zeros(self.num_horizontal_patches*self.num_vertical_patches) and remove flatten.
        patch_id_map = np.zeros((self.num_horizontal_patches, self.num_vertical_patches), dtype=int) # TODO: check if the order correct!
        patch_id_map = patch_id_map.flatten()

        with st.spinner('TRAINING in progress...'):
            unfolded1 = unfolded1.contiguous().numpy().astype('float32')
            unfolded2 = unfolded2.contiguous().numpy().astype('float32')
            if self.index_pretrain:
                if not self.mem.is_trained:
                    self.mem.train(unfolded1)
            # Make sure the resolution is the same or the loop is gonna get wrong!
            # Maybe use numpy.split() method on unfolded1
            # TODO: make sure the indexing is correct!
            st.write(f'SHAPE: {unfolded1.shape}')
            train_progress_bar = st.progress(0)
            for i, pattern1 in enumerate(unfolded1):
                # if row == (self.num_horizontal_patches - 1):
                #     pass

                pattern1 = pattern1.reshape(1, -1)
                pattern2 = unfolded2[i].reshape(1, -1)
                # pattern1 = pattern1.unsqueeze(0)
                # pattern2 = unfolded2[i].unsqueeze(0)

                # pattern1 = pattern1.numpy().astype('float32')
                # pattern2 = pattern2.numpy().astype('float32')

                d1, k1 = self.mem.search(pattern1, 1)
                d2, k2 = self.mem2.search(pattern2, 1)

                k1 = int(k1[0][0])
                k2 = int(k2[0][0])

                # if the pattern1 is not in self.mem add it.
                if d1[0][0] > 0:
                    self.mem.add(pattern1)
                    k1 = self.mem.ntotal - 1
                    self.graph.add_node(k1)
                    self.patches_meta.append(PatchConnections(up=Counter(), down=Counter(), left=Counter(), right=Counter()))
                    # if the pattern2 is not in self.mem2 add it.
                if d2[0][0] > 0:
                    self.mem2.add(pattern2)
                    k2 = self.mem2.ntotal - 1

                mappings = self.pattern_mappings.get(k1)
                if mappings is None:
                    mappings = Counter([k2])
                    self.pattern_mappings[k1] = mappings
                else:
                    self.pattern_mappings[k1].update([k2])
                    # mappings.update([k2])


                patch_id_map[i] = k1


                # self.graph.add_node(k1)
                # column = (i+1) % (self.num_horizontal_patches) == 0
                # row = i//self.num_horizontal_patches
                #
                # if column < (self.num_horizontal_patches - 1):
                #     self.graph.add_edge(k1, k1 + 2)
                # # if row == 0:

                train_progress_bar.progress(i / (len(unfolded1) - 1))

            # column = (i+1) % (self.num_horizontal_patches) == 0
            # row = i//self.num_horizontal_patches
            # for i, id in enumerate(patch_id_map):
            patch_id_map = patch_id_map.reshape((self.num_vertical_patches, self.num_horizontal_patches))

            # # for directed graph
            if True:
                for y in range(self.num_vertical_patches):
                    for x in range(self.num_horizontal_patches):
                        if x < (self.num_horizontal_patches - 1):
                            self.patches_meta[patch_id_map[(y, x)]].right.update([patch_id_map[(y, x + 1)]])
                        if x > 0:
                            self.patches_meta[patch_id_map[(y, x)]].left.update([patch_id_map[(y, x - 1)]])
                        if y < (self.num_vertical_patches - 1):
                            self.patches_meta[patch_id_map[(y, x)]].down.update([patch_id_map[(y + 1, x)]])

                        if y > 0:
                            self.patches_meta[patch_id_map[(y, x)]].up.update([patch_id_map[(y - 1, x)]])
            #
            # # For directed graph
            # if True:
            #     for y in range(self.num_vertical_patches):
            #         for x in range(self.num_horizontal_patches):
            #             if x < (self.num_horizontal_patches - 1):
            #                 self.graph.add_edge(patch_id_map[(y, x)], patch_id_map[(y, x + 1)])
            #             if y < (self.num_vertical_patches - 1):
            #                 self.graph.add_edge(patch_id_map[(y, x)], patch_id_map[(y + 1, x)])

            # st.write(self.patches_meta)
            st.success(f'LEARNED: {self.mem.ntotal}\tpatterns in {time() -  t0} seconds!')
            # st.write(patch_id_map)


image_sizes = [(2**x, 2**x, 3) for x in range(5, 10)]
TRAINING_IMAGE_SIZE = st.sidebar.selectbox(
    'Choose TRAINING image size', options=image_sizes, index=1)
OUTPUT_IMAGE_SIZE = st.sidebar.selectbox(
    'Choose OUTPUT image size', options=image_sizes, index=1)
# IMAGE_SIZE = (64, 64, 3)
# IMAGE_SIZE = (128, 128, 3)
add_selectbox = st.sidebar.selectbox(
    "Use index pretrain?",
    ("YES", "NO"), index=1
)
INDEX_PRETRAIN = True if add_selectbox == "YES" else False

threshold_values = [x/10 for x in range(1, 11)]

THRESHOLD = st.sidebar.selectbox(
    'Choose threshold value', options=threshold_values, index=4)

kernel_sizes = [(x,x) for x in range(1, 33)]
KERNEL_SIZE = st.sidebar.selectbox(
    'Choose kernel size', options=kernel_sizes, index=4)

stride_vals = [x for x in range(1, 11)]
STRIDE = st.sidebar.selectbox(
    'Choose stride value', options=stride_vals, index=0)

padding_vals = [x for x in range(11)]
PADDING = st.sidebar.selectbox(
    'Choose padding value', options=padding_vals, index=10)

num_patches = (int((TRAINING_IMAGE_SIZE[0] - KERNEL_SIZE[0] + 2 * PADDING) / STRIDE) + 1) ** 2 # square
st.sidebar.write(f'PATCHES:\t{num_patches}')

# net = NeuralMem(image_size=TRAINING_IMAGE_SIZE, index_pretrain=INDEX_PRETRAIN, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)

with st.beta_expander("FAST AND ROBUST IMAGE STYLETRANSFER AND COLORIZATION", expanded=True):
    # header1 = st.write('## FAST AND ROBUST IMAGE STYLETRANSFER AND COLORIZATION')
    header2 = st.markdown('#### by providing input and output example image pairs and by using similarity search')
    header3 = st.markdown('##### Transfer the style of images by providing input and output example images.')
    header4 = st.markdown('##### Colorize images by providing black-white or grayscale input and colored output example images(like grayscale photo as input example and colored photo as output example for training)')

# video_file = open('tutorial.webm', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)


col1_1, col1_2 = st.beta_columns(2)
input_ph = st.empty()
train_int_col, train_out_col= st.beta_columns(2)
input_col, output_col = st.beta_columns(2)
rand_input_col, rand_output_col = st.beta_columns(2)


uploaded_inp_example = col1_1.file_uploader("Choose INPUT EXAMPLE for training", type=['png', 'jpg'])
uploaded_out_example = col1_2.file_uploader("Choose OUTPUT EXAMPLE for training", type=['png', 'jpg'])
uploaded_file = input_ph.file_uploader("Choose input image", type=['png', 'jpg']    )

if uploaded_inp_example is not None and uploaded_out_example is not None and uploaded_file is not None:
    net = NeuralMem(image_size=TRAINING_IMAGE_SIZE, index_pretrain=INDEX_PRETRAIN, kernel_size=KERNEL_SIZE,
                    stride=STRIDE, padding=PADDING, threshold=THRESHOLD)
    train_inp_example = preprocess(uploaded_inp_example, image_size=TRAINING_IMAGE_SIZE[0:2], gray_scale=False)
    train_int_col.image(train_inp_example, caption="INPUT EXAMPLE", width=250)
    train_inp_example = torch.tensor(train_inp_example)

    train_out_example = preprocess(uploaded_out_example, image_size=TRAINING_IMAGE_SIZE[0:2], gray_scale=False)
    train_out_col.image(train_out_example, caption="OUTPUT EXAMPLE", width=250)
    train_out_example = torch.tensor(train_out_example)

    # image = preprocess(uploaded_file, image_size=IMAGE_SIZE[0:2], gray_scale=False)
    image = preprocess(uploaded_file, image_size=OUTPUT_IMAGE_SIZE[0:2], gray_scale=False)
    input_col.image(image, width=250, caption='input image')

    net.add(train_inp_example, train_out_example)
    output = net(torch.tensor(image)).numpy()
    output_col.image(output, width=250, caption='output image')

    # st.write(net.graph)
    # net.graph.write_html('graph.html')
    # # components.html(net.graph.html, height)
    # components.html(net.graph.html, height= 600)
    # # st.write(html)
    # breakpoint()

#
# image = torch.rand(IMAGE_SIZE)
# rand_input_col.image(image.numpy(), width=250, caption='random input image')
# output = net(torch.tensor(image)).numpy()
# rand_output_col.image(output, width=250, caption='output image')
