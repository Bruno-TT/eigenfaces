
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

num_components=1
slider_factor=2
glitch_mode=True

images=[]

#find and open the images
image_filepaths = glob("images/*")
for image_filepath in image_filepaths:
    images.append(Image.open(image_filepath))
    
#store the widths and heights
widths, heights = [], []
for image in images:
    width, height = image.size
    widths.append(width)
    heights.append(height)

#size all the images down to the smallest image
mean = lambda iter:int(sum(iter)/len(iter))
size = mean(widths), mean(heights)

# resize the images, and turn them into rows, and print debug statistics
arrays=[]
for image, filename in zip(images, image_filepaths):
    
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
    print("{} - old size: {} - new size: {} - array shape {} - array total size {} - array total size/new size {}".format(filename, old_size, new_size, array_shape, array_total_size, factor))

#merge the images into 1 array
master_array=np.stack(arrays)

#train the model and do pca
model=PCA(n_components=num_components)
vals=model.fit_transform(master_array)

#output the pca vals from the images
for name, val in zip(image_filepaths, vals):print(name, val)

#given pca components, return a PhotoImage
def row_to_image(row, model, size):

    # push the row through the model
    image_array_flattened=model.inverse_transform(row)

    # turn it back into an array
    transformed_array=np.array(image_array_flattened).reshape((size[1], size[0], 4))

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

def button_press_wrapper():
    while 1:display_image_from_sliders()

def randomise_slider(slider, bounds):
    min_=int(min(bounds))
    max_=int(max(bounds))
    val=random.randint(min_, max_)
    slider.set(val)

def randomise_sliders():
    global sliders, sliderbounds
    for slider, bounds in zip(sliders, sliderbounds):randomise_slider(slider, bounds)

root=Tk()
sliderbounds=[(slider_factor*min(vals[:,i]), slider_factor*max(vals[:,i])) for i in range(num_components)]
sliders=[Scale(root, from_=min(bounds), to=max(bounds), orient=HORIZONTAL, length=500) for i,bounds in zip(range(num_components), sliderbounds)]
for slider in sliders:slider.pack()
default_vals_row=random.choice(vals)
for slider, val in zip(sliders, default_vals_row):slider.set(val)

# Button(root, text="ok", command=display_image_from_sliders).pack()
Button(root, text="randomise", command=randomise_sliders).pack()

frame=Frame(root, width=size[0], height=size[1], background='white')
frame.pack()

label=Label(frame)
label.pack()

Thread(target=button_press_wrapper).start()

root.mainloop()