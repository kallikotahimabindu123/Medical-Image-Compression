import numpy as np


def calculate_psnr(original, reconstructed):

    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(original, reconstructed):

    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_x = np.mean(original)
    mu_y = np.mean(reconstructed)

    sigma_x = np.var(original)
    sigma_y = np.var(reconstructed)
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

    if denominator == 0:
        return 1.0

    return numerator / denominator


def entropy(image):

    hist = np.histogram(image.flatten(),256)[0]

    hist = hist/np.sum(hist)

    hist = hist[np.nonzero(hist)]

    return -np.sum(hist*np.log2(hist))


def calculate_metrics(original, reconstructed):

    psnr = calculate_psnr(original, reconstructed)
    ssim = calculate_ssim(original, reconstructed)
    image_entropy = entropy(reconstructed)

    return psnr, ssim, image_entropy

def show_images(original, reconstructed):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Compressed / Reconstructed Image")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')

    plt.show()