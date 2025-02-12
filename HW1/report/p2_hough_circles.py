
import cv2
import numpy as np


def detect_edges(image):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
  - edge_image (2D float array): A heat map where the intensity at each point
      is proportional to the edge magnitude.
  """
  sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
  sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    # Calculate the output dimensions and create an empty output array
  output = np.zeros_like(image, dtype=float)
    
    # Pad the input image with zeros 
  padded_img = np.pad(image, ((1, 1), (1, 1)), 'constant')
    
    # Apply the Sobel masks
  for i in range(1, padded_img.shape[0] - 1):
        for j in range(1, padded_img.shape[1] - 1):
            region = padded_img[i-1:i+2, j-1:j+2]
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)
            output[i-1, j-1] = np.sqrt(gx**2 + gy**2)  # Edge magnitude
  return output


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """
    # Threshold edge image
    thresh_edge_image = np.where(edge_image > edge_thresh, 1, 0)

    # Dimensions of the image
    height, width = edge_image.shape

    # Initialize accumulator array (height x width x number_of_possible_radii)
    accum_array = np.zeros((height, width, len(radius_values)))

    # Indices of the edge pixels
    edge_indices = np.column_stack(np.where(thresh_edge_image > 0))

    # Loop through edge pixels
    for x, y in edge_indices:
        # For each possible radius
        for idx, r in enumerate(radius_values):
            # For each possible circle center for this radius
            for theta in np.arange(0, 360,10):
                a = int(x - r * np.cos(np.deg2rad(theta)))
                b = int(y - r * np.sin(np.deg2rad(theta)))
                
                if a >= 0 and a < height and b >= 0 and b < width:
                    accum_array[a, b, idx] += 1

    return thresh_edge_image, accum_array



def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    # Locations where the accumulator array is above the threshold
    y_coords, x_coords, r_indices = np.where(accum_array > hough_thresh)

    # List of circles to be returned
    circles = [(radius_values[r], y, x) for r, y, x in zip(r_indices, y_coords, x_coords)]

    # Copy of the original image to draw circles
    circle_image = image.copy()

    # Draw the circles using OpenCV
    for r, y, x in circles:
        cv2.circle(circle_image, (x, y), r, (0, 255, 0), thickness=2)

    return circles, circle_image


def main():
    # Load the grayscale image
    image_path = 'data/coins.png'
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    # Detect edges using the Sobel masks
    edges = detect_edges(gray_image)
    
    # Save the result (normalize for visual inspection)
    # edges_normalized = (edges / edges.max() * 255).astype(np.uint8)
    # cv2.imwrite("edges_result.png", edges_normalized)
    edges_normalized = (edges / edges.max() * 255)
    cv2.imwrite("output/edges_result.png", edges_normalized)

    
    # # Optional: compare your results with OpenCV's Sobel
    # sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    # sobel_edges = np.hypot(sobelx, sobely)
    # # sobel_edges_normalized = (sobel_edges / sobel_edges.max() * 255).astype(np.uint8)
    # # cv2.imwrite("edges_sobel_opencv.png", sobel_edges_normalized)
    # cv2.imwrite("output/edges_sobel_opencv.png", sobel_edges)
    # Apply Hough Transform for circle detection
    edge_threshold=108
    radius_range = np.arange(20, 41)  # example range
    thresh_edge_image, accum_array = hough_circles(edges_normalized, edge_threshold, radius_range)
    print("accum_array=",accum_array)
    # Save the thresholded edge image
    cv2.imwrite("output/coins_edges.png", (thresh_edge_image * 255).astype(np.uint8))

    # Find and draw circles
    hough_threshold = (np.max(accum_array)-np.min(accum_array))/2
    circles, circle_image = find_circles(img, accum_array, radius_range, hough_threshold)
    print("circles=",circles)
    # Save the resulting image with circles drawn
    cv2.imwrite("output/coins_circles.png", circle_image)


if __name__ == '__main__':
  #TODO
    main()




