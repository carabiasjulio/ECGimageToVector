import os
import glob
import cv2
import numpy as np
import scipy.signal as signal
import scipy.io as sio
from matplotlib import pyplot as plt
# plt.switch_backend('MacOSX')


# ---------------------------------------------------------------------------------------
# Funciones Auxiliares
# ---------------------------------------------------------------------------------------
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def imadjust(image, gamma=1.0):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def displayImage(image):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    resize = ResizeWithAspectRatio(image, width=1280)  # Resize by width OR
    # resize = ResizeWithAspectRatio(image, height=1280) # Resize by height
    cv2.imshow('resize', resize)
    cv2.waitKey(0)  # Display the image infinitely until any keypressq113' # hit q to exit
    return

# ---------------------------------------------------------------------------------------
# Funciones Principales
# ---------------------------------------------------------------------------------------
def estimacion_periodo_muestreo(imagen, sondeos=5):
    # entrada: imagen , numero lineas para sondear
    # salida: periodo de muestreo estimado
    # funcion:calculo de Ts buscando numero de pixeles en cada cuadr�culo y
    # considerando 0.04 segundos por cuadricula
    nfila, ncol = imagen.shape
    lineas = np.floor(np.random.rand(sondeos)*nfila/2).astype(int)
    periodo = []
    for cont_periodo in np.arange(sondeos):
        linea = imagen[lineas[cont_periodo], :]
        locs, _ = signal.find_peaks(linea, prominence=[0.1])
        # figure(30);findpeaks(linea,'MinPeakProminence',0.1)
        dif_locs = np.diff(locs)
        # dif_locs_ord=sort(dif_locs)
        media_dif = np.mean(dif_locs)
        desv = np.std(dif_locs)
        umbral_periodo = desv/4.
        periodos_validos = dif_locs[np.abs(dif_locs-media_dif) < umbral_periodo]
        if len(periodos_validos) > 0:
            periodo.append(np.mean(periodos_validos))

    periodo_medio = np.mean(periodo)
    Ts = 0.04/periodo_medio
    return Ts


def imagen2vector(imagen):
    # entrada: imagen con ecg
    # salida: vector con ecg
    # funcion: obtener para el vector ecg de la imagen
    # para cada columna de la imagen (instante de tiempo) calcular que valor de
    # amplitud (posicion media de pixeles activos)
    nfila, ncol = imagen.shape
    ecg = np.zeros(ncol)
    ecg[0] = 0  # eliminar, caso primer pixel.
    for cont in np.arange(1, ncol):
        vector = imagen[:, cont]
        inds = indices(vector, lambda x: x == 255)
        if len(inds) > 0:
            mediana_indices = np.median(inds)
            ecg[cont] = mediana_indices
        else:
            ecg[cont] = ecg[cont-1]

    ecg[0] = ecg[1]  # eliminar caso primer pixel
    ecg = nfila-ecg    # ajustar origen de coordenadas

    # MEJORAS: CONSIDERAR TROZOS EN VEZ DE COLUMNAS AISLADAS
    # POST-PROCESAR Y ELIMINAR POCA CONTINUIDAD.
    # PICOS DERIVADA
    return ecg


def elimina_region(imagen, min_area=30):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imagen, connectivity=8)
    # connectedComponentswithStats yields every separated component with information on each of them,
    # such as size the following part is just taking out the background which is also considered a
    # component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_area:
            img2[output == i + 1] = 255
    return img2


def ECG_image_values(file, draw=False):
    img = cv2.bitwise_not(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
    # plt.plot(np.mean(img[:100, :], axis=0))

    # CALCULO DE PERIODO DE MUESTREO.DISTANCIA CUADRICULAS
    kernel = np.ones((100, 1), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(img[:100, :], kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1)
    # --- Metodo Raul ---
    # Ts = estimacion_periodo_muestreo(e_im/255., sondeos=5)
    # --- Metodo Mediana ---
    # e_im = img[:100,:]
    locs, _ = signal.find_peaks(np.median(e_im/255., axis=0), prominence=[0.05])
    # plt.plot(np.median(e_im / 255., axis=0))
    # plt.stem(locs, np.median(e_im / 255., axis=0)[locs], markerfmt='rx')
    dif_locs = np.diff(locs)
    periodo = np.median(dif_locs)
    Ts = 0.04 / periodo

    # AJUSTE NO LINEAL DE CONTRASTE
    im_gray_ad = imadjust(img, gamma=2)

    # BINARIZACION POR BLOQUES
    blur = cv2.GaussianBlur(im_gray_ad, (5, 5), 0)
    _, im_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # PROCESAMIENTO MORFOLOGICO OPENINIG
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)  # DIAMOND KERNEL
    opening = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)

    # PROCESAMIENTO MORFOLOGICO. ELIMINACION AREAS PEQUENAS
    clean_img = elimina_region(opening, min_area=20)

    # CONVERSION IMAGEN A VECTOR
    ecg = imagen2vector(clean_img)
    if draw:
        t = np.arange(0, len(ecg) * Ts, Ts)[:len(ecg)]
        fig, axs = plt.subplots(4)
        fig.suptitle('Ts ' + '{0:.4f}'.format(Ts) + ' - Procesado de ECG para image: ' + os.path.basename(file))
        axs[0].imshow(img, cmap='Greys')
        axs[1].imshow(e_im, cmap='Greys')
        axs[1].stem(locs, 50*np.ones(len(locs)), markerfmt='rx')
        axs[2].imshow(clean_img, cmap='Greys')
        axs[3].plot(t, ecg)
        axs[3].set_xlim([0, np.max(t)])
        plt.show()

    return ecg, Ts


# MAIN FUNCTION
if __name__ == "__main__":
    print('Processing ecgtovector')
    for file in glob.glob("/home/ecgtovector/output/*.jpg"):
        print('Procesing File: ' + file)
        # file = "/Users/carabias/Desktop/ecg/database/ecgc-set1/IMG_0904C.jpg"
        ecg, Ts = ECG_image_values(file, draw=False)
        adict = {}
        adict['ecg'] = ecg
        adict['Ts'] = Ts
        filename, file_extension = os.path.splitext(file)
        sio.savemat(filename + '_python_ecg.mat', adict)


