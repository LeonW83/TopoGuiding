""" Helper method for plotting masks."""
import matplotlib.pyplot as plt

def show_image(img):
    image = img / 2 + 0.5
    image = image.clamp(0, 1)

    if len(image.shape) == 4:
        image = image[0]

    image = image.cpu().detach().numpy()

    if image.shape[0] == 1:
        image = image.squeeze(0)
        plt.imshow(image, vmin=0, vmax=1, cmap='gray')
    else:
        image = image.transpose(1, 2, 0)
        plt.imshow(image)

    plt.axis('off')
    plt.show()
