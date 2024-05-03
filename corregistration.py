from matplotlib import pyplot as plt, animation
import matplotlib
import numpy as np
import pydicom
import glob
import cv2
import scipy
import os


def find_landmarks(image):
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecta esquinas utilizando el algoritmo Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Convierte las coordenadas de las esquinas a enteros
    corners = np.int(corners)

    # Extrae las coordenadas de las esquinas como una lista de tuplas
    landmarks = [tuple(corner.ravel()) for corner in corners]

    return np.array(landmarks)

if __name__ == '__main__':
    reference_path = "part_2/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm"
    reference = pydicom.read_file(reference_path)
    img_reference = reference.pixel_array

    input_path = "part_2/RM_Brain_3D-SPGR"
    # load the DICOM files
    input_slices = []
    # Use glob to get all the files in the folder
    files = glob.glob(input_path + '/*')
    for fname in files:
        print("loading: {}".format(fname))
        input_slices.append(pydicom.dcmread(fname))
    print("CT slices loaded! \n")

    # Sort according to ImagePositionPatient header, based on the 3rd component (which is equal to "SliceLocation")
    print("Sorting slices...")
    input_slices = sorted(input_slices, key=lambda ds: float(ds.ImagePositionPatient[2]))
    print("Slices sorted!")

    # pixel aspects, assuming all slices are the same
    ps = input_slices[0].PixelSpacing
    ss = input_slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    # create 3D array
    img_shape = list(input_slices[0].pixel_array.shape)
    img_shape.append(len(input_slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(input_slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    print("shape input img3d: " + str(img3d.shape))
    print("shape reference img" + str(img_reference.shape))
    img_reference = np.transpose(img_reference, (0,2,1))
    print("shape reference img" + str(img_reference.shape))

    #landmarks = find_landmarks(img3d[:, :, img_shape[2] // 2])  # Encuentra los puntos de referencia
    #print(landmarks)

    def rigid_registration_3d(image_fixed, image_moving, learning_rate=0.1, num_iterations=100):
        # Inicializar la transformación con una matriz de identidad
        transform_matrix = np.eye(4)

        # Extraer las dimensiones de las imágenes
        _, _, num_slices = image_fixed.shape

        # Definir la función de costo (puedes usar la suma de las diferencias cuadradas)
        def cost_function(transform_matrix):
            registered_image = apply_transform(image_moving, transform_matrix)
            cost = np.sum((image_fixed - registered_image) ** 2)
            return cost

        # Definir el gradiente de la función de costo
        def cost_gradient(transform_matrix):
            epsilon = 1e-5
            grad = np.zeros_like(transform_matrix)
            for i in range(6):
                delta = np.zeros_like(transform_matrix)
                delta[i // 4, i % 4] = epsilon
                cost_plus = cost_function(transform_matrix + delta)
                cost_minus = cost_function(transform_matrix - delta)
                grad[i // 4, i % 4] = (cost_plus - cost_minus) / (2 * epsilon)
            return grad

        # Optimizar la transformación usando descenso del gradiente
        for _ in range(num_iterations):
            grad = cost_gradient(transform_matrix)
            transform_matrix -= learning_rate * grad

        return transform_matrix


    def apply_transform(image, transform_matrix):
        # Aplicar la transformación a la imagen movible
        num_slices = image.shape[2]
        registered_image = np.zeros_like(image)
        for i in range(num_slices):
            # Aplicar la transformación a cada píxel de la imagen
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    # Obtener las coordenadas transformadas
                    new_coords = np.dot(transform_matrix[:3, :3], [x, y, i]) + transform_matrix[:3, 3]
                    new_x, new_y, new_z = new_coords.astype(int)
                    # Verificar si las nuevas coordenadas están dentro de los límites de la imagen
                    if 0 <= new_x < image.shape[0] and 0 <= new_y < image.shape[1] and 0 <= new_z < num_slices:
                        registered_image[new_x, new_y, new_z] = image[x, y, i]
        return registered_image

    # Uso:
    # image_fixed y image_moving son imágenes 3D representadas como matrices NumPy
    # Se puede obtener la transformación resultante y aplicarla a la imagen movible
    transform_matrix = rigid_registration_3d(img_reference, img3d)
    registered_image = apply_transform(img3d, transform_matrix)

#factor de escala (mm), skimage transform o scipy y numpy