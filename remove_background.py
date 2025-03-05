from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

from helper_classes import Rectangle



# MARK: Helper Functions



def read_image(image_path: str) -> cv2.typing.MatLike:
    """
    This reads in an image into a cv2 image from a filepath
    """
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Failed to read image at path: {image_path}")
    return raw_image



def get_pixelwise_outliers(gray_image: cv2.typing.MatLike):
    """
    This analyzes each pixel in the image on an individual basis, and uses the histogram of pixel intensity to determine the intensity outliers in the image.
    This is necessary in addition to the regionwise analysis because in some images, the background is made up of multiple similar-intensity pixels.
    For example, many images will have a black looking background, but the individual pixels will be speckled with some having intensity 0 and others having intensity 4.
    These different intensities won't necessarily be clustered together either, so its unlikely in these cases that the region analyzer will pick up on those pixels.
    """
    # Compute the pixel intensity distribution
    pixel_values, pixel_counts = np.unique(gray_image, return_counts = True)

    # Calculate the Z-scores of the pixel counts
    z_scores = zscore(pixel_counts)

    # This value represents the number of standard deviations of the normal curve that will threshold an 'outlier'.
    threshold = 3
    outliers = pixel_values[z_scores > threshold]
    """
    In some images, the distribution of pixels is very skewed to a range of just a few values. In these images, you can have half the pixel values falling in "outlier" pixel intensities.
    For these images, we don't want to zero out all those pixels, because then half or more of the image will be erased, making finding a bounding box far more difficult.
    Thus, we set a limit of 3 values to be erased from the image, taking the 3 most common pixel intensities as those values.
    This should be enough tho remove the background from images, as most images have only 1-2 distinct pixel intensities as their background.
    """
    outliers = np.sort(outliers)[-3:]

    return outliers, z_scores, pixel_values, pixel_counts



def get_regionwise_outliers(gray_image: cv2.typing.MatLike):
    """
    This analyzes the connected components in an image and determines which intensity values have the largest connected components.
    If any intensity's largest area is an outlier in the set, it's likely because the intensity corresponds to one of the image's background intensities.
    Thus, we return these intensities so they can be zeroed out in the image.
    """ 
    # Find unique intensities in the image
    unique_intensities = np.unique(gray_image)

    # This store the maximum connected component area for each intensity
    intensity_areas = []

    # Analyze connected components for each intensity, one by one.
    for intensity in unique_intensities:
        # Threshold the image to get only the current intensity
        binary_image = (gray_image == intensity).astype(np.uint8) * 255

        # This finds all the connected components in the binary image. We use 4-connectivity (ie components are only connected if they touch on one of the sides or the bottom or top) because we are looking for basically rectangular regions.
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity = 4)

        # The area of the largest connected component
        if num_labels > 1:  # Num_labels will always be > 0 because of the background. In this case, we use 1 to ensure that there is at least one nonzero connected component other than the background.
            max_area = np.max(stats[1:, cv2.CC_STAT_AREA]) # We use 1 rather than 0 to skip the first component which is the background
        else:
            max_area = 0

        intensity_areas.append(max_area)

    # Convert lists to numpy arrays for statistical analysis
    intensity_areas = np.array(intensity_areas)

    # Identify outliers using z-score
    z_scores = zscore(intensity_areas)
    outlier_indices = np.where(z_scores > 3)  # You can adjust the threshold as needed

    # Get the outlier intensity values
    outlier_intensities = unique_intensities[outlier_indices]
    """
    In some images, there are many connected components clustered together within the actual ultrasound image. In these images, many connected regions will appear as outliers.
    Since zeroing all these intensities would potentially remove large percentages of the ultrasound image, it would make the next steps more difficult.
    Thus, we limit the number of outliers to zero out to 3, since typically, backgrounds aren't made of more than 1-2 distinct intensities.
    """
    outlier_intensities = np.sort(outlier_intensities)[-3:]

    return outlier_intensities, z_scores, unique_intensities, intensity_areas



def remove_outlier_intensities(image: cv2.typing.MatLike, show_intensity_histogram: bool = False) -> cv2.typing.MatLike:
    """
    This takes in an image and analyzes the intensity distribution. When it finds outliers, it typically means these value correspond to operator or radiologist placed annotations.
    Often, the background of the image will all be a certain shade of gray, and letters burnt in to the image will all be a certain intensity.
    When such anomolous intensities are found, we set all the pixels with such values to 0. Though this will affect some of the natural ultrasound image, it's okay, as these changes are not permanent; they are only used in the background removal pipeline.
    Once the pipeline determines where the actual image is, we impose that region on the original image, preserving those pixel values.
    Ultimately, only a few pixel values will end up being removed, so the actual ultrasound image that continues past this function at large will look similar.

    Note that a different method of removing such 'high mode' intensities would be to set the thresholding value based on the mode (so if the image background had an intensity of 8, everthing 8 or below would go to zero).
    This method was found to be inferior, because ultimately, some images have very light backgrounds (up to 128 even), so you had to either choose whether to threshold half the intensity values, removing a lot of the real image, or set a maximum thresholding value, making the pipeline ineffective for images with light backgrounds.
    """
    # Convert the image to grayscale if it has three channels.
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    pixelwise_outliers, pixelwise_z_scores, pixelwise_pixel_values, pixelwise_pixel_counts = get_pixelwise_outliers(gray_image = gray_image)
    regionwise_outliers, regionwise_z_scores, regionwise_pixel_values, regionwise_areas = get_regionwise_outliers(gray_image = gray_image)

    # The outliers from each different search method are combined and the concatenated result is then used to remove pixels from the image.
    combined_outliers = np.unique(np.concatenate((pixelwise_outliers, regionwise_outliers)))
    # This mask corresponds to every pixel where the intensity is in outliers
    mask = np.isin(gray_image, combined_outliers)

    outliers_removed_image = gray_image.copy()
    outliers_removed_image[mask] = 0


    if show_intensity_histogram:
        _, axs = plt.subplots(3, 3, figsize=(14, 9))  # Correcting the figure size for better spacing

        # Plotting the grayscale image
        axs[0, 0].imshow(gray_image, cmap='gray', vmin=0, vmax=255)
        axs[0, 0].set_title('Grayscale Image')
        axs[0, 0].axis('off')

        # Plotting the pixelwise outliers removed image
        pixelwise_outliers_mask = np.isin(gray_image, pixelwise_outliers)
        pixelwise_outliers_removed_image = gray_image.copy()
        pixelwise_outliers_removed_image[pixelwise_outliers_mask] = 0
        axs[0, 1].imshow(pixelwise_outliers_removed_image, cmap='gray', vmin=0, vmax=255)
        axs[0, 1].set_title('Pixelwise Outliers Removed Image')
        axs[0, 1].axis('off')

        # Plotting the regionwise outliers removed image
        regionwise_outliers_mask = np.isin(gray_image, regionwise_outliers)
        regionwise_outliers_removed_image = gray_image.copy()
        regionwise_outliers_removed_image[regionwise_outliers_mask] = 0
        axs[0, 2].imshow(regionwise_outliers_removed_image, cmap='gray', vmin=0, vmax=255)
        axs[0, 2].set_title('Regionwise Outliers Removed Image')
        axs[0, 2].axis('off')

        # Plotting the final image
        axs[1, 0].imshow(outliers_removed_image, cmap='gray', vmin=0, vmax=255)
        axs[1, 0].set_title('Final Image')
        axs[1, 0].axis('off')

        # Plotting the pixelwise z-values
        axs[1, 1].bar(pixelwise_pixel_values, pixelwise_z_scores, width=1.0)
        axs[1, 1].set_title('Pixelwise Z Values')
        axs[1, 1].set_xlabel('Pixel Value')
        axs[1, 1].set_ylabel('Z Score')
        axs[1, 1].grid(True)

        # Plotting the regionwise z-values
        axs[1, 2].bar(regionwise_pixel_values, regionwise_z_scores, width=1.0)
        axs[1, 2].set_title('Regionwise Z Values')
        axs[1, 2].set_xlabel('Pixel Value')
        axs[1, 2].set_ylabel('Z Score')
        axs[1, 2].grid(True)

        # Plotting the pixelwise pixel intensities
        axs[2, 1].bar(pixelwise_pixel_values, pixelwise_pixel_counts, width=1.0, edgecolor='black')
        axs[2, 1].set_title('Pixelwise Pixel Values')
        axs[2, 1].set_xlabel('Pixel Value')
        axs[2, 1].set_ylabel('Count')
        axs[2, 1].grid(True)

        # Plotting the regionwise pixel intensity areas
        axs[2, 2].bar(regionwise_pixel_values, regionwise_areas, width=1.0, edgecolor='black')
        axs[2, 2].set_title('Regionwise Intensity Areas')
        axs[2, 2].set_xlabel('Pixel Value')
        axs[2, 2].set_ylabel('Area')
        axs[2, 2].grid(True)

        # Displaying the list of outliers
        pixelwise_outliers_text = '\n'.join(map(str, pixelwise_outliers))
        axs[1, 1].text(0.95, 0.95, f'Outliers:\n{pixelwise_outliers_text}', verticalalignment='top', horizontalalignment='right', transform=axs[1, 1].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        regionwise_outliers_text = '\n'.join(map(str, regionwise_outliers))
        axs[1, 2].text(0.95, 0.95, f'Outliers:\n{regionwise_outliers_text}', verticalalignment='top', horizontalalignment='right', transform=axs[1, 2].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        plt.tight_layout()
        plt.show()


    return outliers_removed_image



def get_binary_image(outlier_removed_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    This takes in an image and thresholds it based on the mode of the image.
    """ 
    # Apply a binary threshold to the image
    _, binary = cv2.threshold(outlier_removed_image, 0, 255, cv2.THRESH_BINARY)

    return binary



def close_image(binary_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    This applies the image closure operation. This operation is idempotent, so there is no use in repeating it a given number of times.
    This can be used to join together all the 'speckles' in an image after thresholding, particularly in hypoechoic areas where there are many disconnected speckles, but no real connected region.
    """
    # Define the kernel
    kernel = np.ones((7, 7), dtype = np.uint8)
    
    # Apply the closing operation (dilation followed by erosion)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return closed_image



def erode_image(binary_image: cv2.typing.MatLike, iterations: int) -> cv2.typing.MatLike:
    """
    This takes in an image, and erodes it a given number of times.
    This can be used to remove thin annotations and markings near to boundary of the image.
    """
    # The kernal looks like
    # 0 1 0
    # 1 1 1
    # 0 1 0
    # Thus, it keeps only pixels that are attatched to the center of the image.
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)

    eroded_image = cv2.erode(binary_image, kernel, iterations = iterations)

    return eroded_image



def dilate_image(binary_image: cv2.typing.MatLike, iterations: int) -> cv2.typing.MatLike:
    """
    This takes in an image, and dilates it a given number of times.
    If applied after erosision, this can restore the general shape of the image before erosion.
    """
    # The kernal looks like
    # 0 1 0
    # 1 1 1
    # 0 1 0
    # Thus, it adds pixels that are adjacent to any neighboring 1.
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)

    dilated_image = cv2.dilate(binary_image, kernel, iterations = iterations) 

    return dilated_image




def get_largest_contour(binary_image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    This takes in a binary image and returns a contour that encloses all contours that are similar in size 
    to the largest contour. This helps capture multiple significant regions that may be disconnected (like two regions separated by an artery).
    """
    # Find contours in the binary image
    all_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not all_contours:
        return np.array([[[0,0]]], dtype=np.int32)  # Return minimal contour if none found
        
    # Get areas of all contours
    areas = [cv2.contourArea(c) for c in all_contours]
    max_area = max(areas)
    
    # Find contours with area at least 50% of the largest
    significant_contours = [c for c, area in zip(all_contours, areas) 
                          if area > 0.5 * max_area]
    
    # Combine all points from significant contours
    all_points = np.vstack([c.squeeze() for c in significant_contours])
    
    # Find convex hull around all significant contours
    hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
    
    return hull



def make_contour_convex(contour: cv2.typing.MatLike):
    """
    The contour drawn around the binary image can be complex, and can weave into the image.
    It is imparitive to "blow up" this contour into a convex shape, so that when the data outside of the contour is zeroed out, it doesn't inadvertintly zero out some of the image that fell outside the contour.
    """
    hull = cv2.convexHull(contour)
    return hull



def apply_contour_mask(image: cv2.typing.MatLike, contour: cv2.typing.MatLike):
    """
    This function zeros out all the image pixels outside of the contour.
    """
    # Create a mask with the same dimensions as the image, initialized to zero
    mask = np.zeros_like(image)

    # Fill the contour on the mask with white
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness = cv2.FILLED)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, mask)
    
    return result



def get_contour_bounding_box(contour: cv2.typing.MatLike):
    """
    This takes in a contour and returns the smallest rectangle containing it.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

    

def crop_image_to_contour_box(image: cv2.typing.MatLike, contour: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, Rectangle]:
    """
    This takes in an image and crops it to the box enclosing the contour. It also zeroes out all the pixels outside of the contour.
    """
    masked_image = apply_contour_mask(image = image, contour = contour)

    # Get bounding box coordinates
    x, y, w, h = get_contour_bounding_box(contour)
    
    # Crop the image to the bounding box
    masked_cropped_image = masked_image[y: y + h, x : x + w]
    
    # Define the cropping box as a Rectangle
    cropping_box = Rectangle(start_x = x, start_y = y, end_x = x + w, end_y = y + h)
    
    return masked_cropped_image, cropping_box






# MARK: Visualization and Display



def get_contour_overlayed_image(contour, image):
    """
    This takes in an image and a contour, and overlays the contour onto the image
    """
    # Create a copy of the original image to draw the contour on
    contour_image = image.copy()

    # This draws the 0th contour from the provided list of contours (in this case the list has one element) and draws it onto contour_image.
    cv2.drawContours(contour_image, [contour], 0, (0, 255, 0), 4)

    return contour_image



def display_background_removal(raw_image, outlier_removed_image, binary_image, closed_image, eroded_image, dilated_image, contour_overlayed_image, convex_contour_overlayed_image, final_image):
    """
    This displays the process of background removal for a given image. It allows the process to be visually adjusted before being applied to all the images.
    """
    num_image_rows = 3
    num_image_columns = 3

    plt.figure(figsize=(14, 9))

    plt.subplot(num_image_rows, num_image_columns, 1)
    plt.axis('off')
    plt.title('Raw Image')
    plt.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    plt.axes

    plt.subplot(num_image_rows, num_image_columns, 2)
    plt.axis('off')
    plt.title('Outlier Removed Image')
    plt.imshow(cv2.cvtColor(outlier_removed_image, cv2.COLOR_BGR2RGB))
    plt.axes

    plt.subplot(num_image_rows, num_image_columns, 3)
    plt.axis('off')
    plt.title('Binary Image')
    plt.imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))
    plt.axes

    plt.subplot(num_image_rows, num_image_columns, 4)
    plt.axis('off')
    plt.title('Closed Image')
    plt.imshow(cv2.cvtColor(closed_image, cv2.COLOR_BGR2RGB))
    plt.axes

    plt.subplot(num_image_rows, num_image_columns, 5)
    plt.axis('off')
    plt.title('Eroded Image')
    plt.imshow(cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB))

    plt.subplot(num_image_rows, num_image_columns, 6)
    plt.axis('off')
    plt.title('Dilated Image')
    plt.imshow(cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB))

    plt.subplot(num_image_rows, num_image_columns, 7)
    plt.axis('off')
    plt.title('Original Contoured Image')
    plt.imshow(cv2.cvtColor(contour_overlayed_image, cv2.COLOR_BGR2RGB))

    plt.subplot(num_image_rows, num_image_columns, 8)
    plt.axis('off')
    plt.title('Convex Contoured Image')
    plt.imshow(cv2.cvtColor(convex_contour_overlayed_image, cv2.COLOR_BGR2RGB))

    plt.subplot(num_image_rows, num_image_columns, 9)
    plt.axis('off')
    plt.title('Final Image')
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

    plt.show()








# MARK: Background Removal



def remove_background(
    image: str | cv2.typing.MatLike, 
    erosion_iterations = 5, 
    dilation_iterations = 5, 
    display_removal: bool = False
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    This function takes in a raw image (either as path or cv2 image) and removes and crops the background.
    It does this by thresholding, then eroding a given number of times, then dilating a given number of times, 
    then finding the biggest contour and drawing a rectangle around it as the crop zone.
    """
    # Handle input as either path or image
    if isinstance(image, str):
        raw_image = read_image(image)
    else:
        raw_image = image

    outlier_removed_image = remove_outlier_intensities(image = raw_image, show_intensity_histogram = False)
    binary_image = get_binary_image(outlier_removed_image)
    closed_image = close_image(binary_image = binary_image)
    eroded_image = erode_image(binary_image = closed_image, iterations = erosion_iterations)
    dilated_image = dilate_image(binary_image = eroded_image, iterations = dilation_iterations)
    largest_contour = get_largest_contour(dilated_image)
    convex_contour = make_contour_convex(contour = largest_contour)

    # Crop the image to the contour bounding box
    cropped_image, cropping_box = crop_image_to_contour_box(raw_image, convex_contour)

    # Display removal process if requested
    if display_removal:
        contour_overlayed_image = get_contour_overlayed_image(largest_contour, raw_image)
        convex_contour_overlayed_image = get_contour_overlayed_image(contour = convex_contour, image = raw_image)
        display_background_removal(
            raw_image = raw_image, 
            outlier_removed_image = outlier_removed_image, 
            binary_image = binary_image, 
            closed_image = closed_image, 
            eroded_image = eroded_image, 
            dilated_image = dilated_image, 
            contour_overlayed_image = contour_overlayed_image, 
            convex_contour_overlayed_image = convex_contour_overlayed_image, 
            final_image = cropped_image
        )

    return cropped_image, cropping_box



if __name__ == "__main__":
    image_path = "/Users/spencermarx/Downloads/Databases-Muscular-DeepACSA-Source A-Data-rectus_images-rectus_img_108.tif"
    cropped_image, cropping_box = remove_background(image_path, erosion_iterations = 8, dilation_iterations = 8, display_removal = True)
    print(cropping_box)