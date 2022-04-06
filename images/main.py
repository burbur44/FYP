from simple_unet_model import simple_unet_model
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import tensorflow as tf
from sklearn.model_selection import train_test_split

# read in the two tiff stacks
large_image_stack = tiff.imread('20/Images/images.tif')
large_mask_stack = tiff.imread('20/Masks/masks.tif')

# use patchify to segment the images and the masks
all_img_patches = []
for img in range(large_image_stack.shape[0]):
    print(large_image_stack.shape)
    print(img)

    large_image = large_image_stack[img]

    patches_img = patchify(large_image, (256, 256), step=256)  # Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            single_patch_img = (single_patch_img.astype('float32')) / 255.

            all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)
images = np.expand_dims(images, -1)

all_mask_patches = []
for img in range(large_mask_stack.shape[0]):
    print(img)  # just stop here to see all file names printed

    large_mask = large_mask_stack[img]

    patches_mask = patchify(large_mask, (256, 256), step=256)  # Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]
            single_patch_mask = single_patch_mask / 255.

            all_mask_patches.append(single_patch_mask)

masks = np.array(all_mask_patches)
masks = np.expand_dims(masks, -1)

print(images.shape)
print(masks.shape)
print("Pixel values in the mask are: ", np.unique(masks))

# create test and train set, with a .30 split and using random state to be able to reproduce results
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.30, random_state=0)
print(X_train.shape)
print(X_test.shape)

IMG_HEIGHT = images.shape[1]
IMG_WIDTH = images.shape[2]
IMG_CHANNELS = images.shape[3]


def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


model = get_model()

# randomly print out an image and assosciated mask to check if they match
import random
import numpy as np

image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

# New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks.
# This gives a binary mask rather than a mask with interpolated values.
seed = 24
from keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(shear_range=0,
                         width_shift_range=0,
                         height_shift_range=0,
                         validation_split=0.2)

mask_data_gen_args = dict(shear_range=0,
                          width_shift_range=0,
                          height_shift_range=0,
                          validation_split=0.2,
                          preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(
                              x.dtype))  # Binarize the output again.

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(X_train, augment=True, seed=seed)

image_generator = image_data_generator.flow(X_train, seed=seed)
valid_img_generator = image_data_generator.flow(X_test, seed=seed)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, seed=seed)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)


def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

checkpointer = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose=2, save_best_only=True)

# create callbacks to enable early stopping and to use tensorboard
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]

batch_size = 16
steps_per_epoch = 3 * (len(X_train)) // batch_size
# fit the model
history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=steps_per_epoch,
                    validation_steps=steps_per_epoch, epochs=100, callbacks=callbacks)

# plot the training and validation accuracy and loss for each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# calculation IOU
y_pred = model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)

# Test prediction on images


test_img_number = len(X_test)
for i in range(test_img_number):
    test_img = X_test[i]
    ground_truth = y_test[i]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.2).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')

    plt.show()

# IoU score is:  0.8718063648588077
