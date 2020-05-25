import gc
from imageio.core.util import Array
import numpy as np
from glob import glob
import imageio
import time
from sklearn.decomposition import PCA
import io
from tkinter import Tk, Label, Scale, Button, HORIZONTAL, Frame
from threading import Thread
import random
import PIL
from PIL import Image
from PIL import ImageTk

num_components=10
slider_factor=1
max_images=1000
display_presets=False
display_sliders=True
glitch_mode=False

images=[]

#find and open the images
image_filepaths=glob("images/*")
if max_images!=True:
    image_filepaths = random.sample(image_filepaths,max_images)
for image_filepath in image_filepaths:
    images.append(Image.open(image_filepath))
num_images=len(image_filepaths)
del image_filepaths
    
#store the widths and heights
widths, heights = [], []
for image in images:
    width, height = image.size
    widths.append(width)
    heights.append(height)

#size all the images down to the smallest image
mean = lambda iter:int(sum(iter)/len(iter))
size = mean(widths), mean(heights)
del widths, heights

# resize the images, and turn them into rows, and print debug statistics
arrays=[]
for image in images:
    
    old_size=image.size
    image_resized=image.resize(size)
    new_size=image_resized.size
    new_size_total=new_size[0]*new_size[1]

    array=np.array(image_resized)
    array_shape=array.shape
    array=array.flatten()
    arrays.append(array)

    
    array_total_size=array_shape[0]*array_shape[1]
    factor=array_total_size/new_size_total
    # print("old size: {} - new size: {} - array shape {} - array total size {} - array total size/new size {}".format(filename, old_size, new_size, array_shape, array_total_size, factor))
del images

#merge the images into 1 array
master_array=np.stack(arrays)
del arrays

#train the model and do pca
model=PCA(n_components=num_components)

gc.collect()

vals=model.fit_transform(master_array)

#output the pca vals from the images
# for name, val in zip(image_filepaths, vals):print(name, val)

#given pca components, return a PhotoImage
def row_to_image(row, model, size):

    # push the row through the model
    image_array_flattened=model.inverse_transform(row)

    # turn it back into an array
    transformed_array=np.array(image_array_flattened).reshape((size[1], size[0], 3))

    #turn it into an imageio image
    new_image=Array(transformed_array)

    #save as a temp file
    if glitch_mode:new_image=new_image.astype(np.uint8)
    imageio.imwrite("temp/TEMP.png", new_image)

    # time.sleep(1)

    #return it as a PhotoImage object
    return ImageTk.PhotoImage(file="temp/TEMP.png")


def display_image_from_sliders():
    global sliders, model, size, label
    global photoimage #do not touch - otherwise gc will delete image from memory
    
    vals=[slider.get() for slider in sliders]
    row=np.array([vals])
    photoimage=row_to_image(row, model, size)

    label.config(image=photoimage)

def display_updater_wrapper():
    global sliders, old_vals
    while 1:
        new_vals=[slider.get() for slider in sliders]
        # print(new_vals)
        if new_vals!=old_vals:
            display_image_from_sliders()
            old_vals=new_vals

def randomise_slider(slider, bounds):
    min_=int(min(bounds))
    max_=int(max(bounds))
    val=random.randint(min_, max_)
    slider.set(val)

def randomise_sliders():
    global sliders, sliderbounds
    for slider, bounds in zip(sliders, sliderbounds):randomise_slider(slider, bounds)

def set_sliders_to_val_row(sliders, vals_row):
    for slider, val in zip(sliders, vals_row):slider.set(val)

def multiply_bounds_from_mean(bounds, factor):
    midpoint=mean(bounds)
    distance=midpoint-bounds[0]
    return (midpoint-factor*distance, midpoint+factor*distance)


root=Tk()

component_bounds=[(min(vals[:,i]), max(vals[:,i])) for i in range(num_components)]
sliderbounds=[multiply_bounds_from_mean(bounds, slider_factor) for bounds in component_bounds]
sliders=[Scale(root, from_=min(bounds), to=max(bounds), length=500) for i,bounds in zip(range(num_components), sliderbounds)]

for n,slider in enumerate(sliders):
    if display_sliders:
        slider.grid(row=0, column=num_images+n, rowspan=3)

default_vals_row=random.choice(vals)
set_sliders_to_val_row(sliders, default_vals_row)


# Button(root, text="ok", command=display_image_from_sliders).pack()
Button(root, text="randomise", command=randomise_sliders).grid(row=2, column=0, columnspan=num_images)

frame=Frame(root, width=size[0], height=size[1], background='white')
frame.grid(row=0, column=0, columnspan=num_images)

label=Label(frame)
label.pack()

if display_presets:
    for n,row in enumerate(vals):
        func=lambda row=row:set_sliders_to_val_row(sliders, row)
        Button(text="preset #{}".format(str(n+1)), command=func).grid(row=1, column=n)

old_vals=None
Thread(target=display_updater_wrapper).start()

root.mainloop()