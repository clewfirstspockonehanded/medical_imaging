import numpy as np
import cv2
import pydicom
from skimage import measure, morphology
from tensorflow.keras import backend as K
import random


def load_mask_instance(row):
    """Load instance masks for the given annotation row. Masks can be different types,
    mask is a binary true/false map of the same size as the image.
    """

    mask = np.zeros((row.height, row.width), dtype=np.uint8)

    annotation_mode = row.annotationMode
    # print(annotation_mode)

    if annotation_mode == "bbox":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        w = int(row["data"]["width"])
        h = int(row["data"]["height"])
        mask_instance = mask[:, :].copy()
        cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
        mask[:, :] = mask_instance

    # FreeForm or Polygon
    elif annotation_mode == "freeform" or annotation_mode == "polygon":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:, :].copy()
        cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
        mask[:, :] = mask_instance

    # Line
    elif annotation_mode == "line":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:, :].copy()
        cv2.polylines(mask_instance, np.int32([vertices]), False, (255, 255, 255), 12)
        mask[:, :] = mask_instance

    elif annotation_mode == "location":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        mask_instance = mask[:, :].copy()
        cv2.circle(mask_instance, (x, y), 7, (255, 255, 255), -1)
        mask[:, :] = mask_instance

    elif annotation_mode is None:
        print("Not a local instance")

    return mask


def load_scan(paths):
    slices = [pydicom.read_file(path) for path in paths]
    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=True)
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    # Improvement: Pick multiple background labels from around the  patient
    # More resistant to “trays” on which the patient lays cutting the air around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == "Model" or layer_type == "Functional":
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum(
        [K.count_params(p) for p in model.non_trainable_weights]
    )

    number_size = 4.0
    if K.floatx() == "float16":
        number_size = 2.0
    if K.floatx() == "float64":
        number_size = 8.0

    total_memory = number_size * (
        batch_size * shapes_mem_count + trainable_count + non_trainable_count
    )
    gbytes = np.round(total_memory / (1024.0**3), 3) + internal_model_mem_count
    return gbytes


def batch_generator(batch_size, preprocess_input, patient_pixels, masks):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            image_list.append(patient_pixels)
            mask_list.append(masks)

        image_list = np.array(image_list, dtype=np.float32)
        image_list = preprocess_input(image_list)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0
        # print(image_list.shape, mask_list.shape)
        yield image_list, mask_list


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def grey_to_rgb_3d(images):
    result = []
    for i in images:
        if i.min() < 0:
            i = (normalize_data(i) * 255).astype(np.uint8)
        result.append(cv2.cvtColor(i, cv2.COLOR_GRAY2RGB))
    return np.array(result)
