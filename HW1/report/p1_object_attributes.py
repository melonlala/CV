
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  binary_image = np.where(gray_image >= thresh_val, 255, 0).astype(np.uint8)
  return binary_image

def label(binary_image):
  # TODO
  label = 50
  labeled_image = np.zeros_like(binary_image, dtype=int)
  new_labeled_image = np.zeros_like(binary_image, dtype=int)
  equivalences = {}

  # First Pass
  for i in range(binary_image.shape[0]):
      for j in range(binary_image.shape[1]):
          if binary_image[i, j] == 255:
              neighbors = [(i-1,j-1),(i-1, j), (i-1,j+1),(i, j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1) ]
              # neighbors = [(i-1,j), (i,j-1), (i,j+1), (i+1,j)]
              # neighbors=[(i-1,j-1), (i-1, j), (i, j-1)]
              neighbor_labeled_image = [labeled_image[x, y] for x, y in neighbors if 0 < x < binary_image.shape[0] and 0 < y < binary_image.shape[1] and labeled_image[x, y] > 0]
              # if len(neighbor_labeled_image)==3:
              #     print("wrong")
              if not neighbor_labeled_image:
                  labeled_image[i, j] = label
                  label  = label+1
                  # continue
              # if len(neighbor_labeled_image)==1:
              #     labeled_image[i,j]=neighbor_labeled_image[0]
              else:
                  min_label = min(neighbor_labeled_image)
                  labeled_image[i, j] = min_label
                  # equivalences[neighbor_labeled_image[1]]=neighbor_labeled_image[0]
                  for lbl in neighbor_labeled_image:
                      if lbl != min_label:
                          if min_label in equivalences:
                              equivalences[lbl] = equivalences[min_label]
                          else:
                              equivalences[lbl] = min_label

  # Second Pass
  for i in range(binary_image.shape[0]):
      for j in range(binary_image.shape[1]):
          while labeled_image[i, j] in equivalences:
              labeled_image[i, j] = equivalences[labeled_image[i, j]]
  return labeled_image

def get_attribute(labeled_image):
  # TODO
  labels = np.unique(labeled_image)
  attribute_list = []
  num = 0
  for lbl in labels:
      if lbl == 0:  # Background
          continue
      
      # Mask for the current object
      mask = (labeled_image == lbl)
      tmp_img = tmp = np.where(mask==1,255,0).astype(np.uint8)
      path = 'output/object_{}.png'.format(num)
      # cv2.imwrite(path, tmp_img)
      num = num+1
      # Get pixel coordinates for the object
      real_y, x = np.nonzero(mask)
      y = np.zeros_like(real_y)
      for i in range(len(real_y)):
         y[i]=labeled_image.shape[0]-real_y[i]
      # Centroid/Position
      x_centroid = x.mean()
    #   y_centroid = labeled_image.shape[0]-y.mean()
      y_centroid = y.mean()
      # Moments for orientation
      mu20 = np.sum((x - x_centroid)**2)
      mu02 = np.sum((y - y_centroid)**2)
      mu11 = np.sum((x - x_centroid) * (y - y_centroid))
      
      # Orientation
      orientation = 0.5 * np.arctan2(2*mu11, mu20 - mu02)
    #   orientation = np.degrees(orientation)

      # Roundedness
      E_min = mu20 * np.sin(orientation)**2 - 2*mu11 * np.sin(orientation) * np.cos(orientation) + mu02 * np.cos(orientation)**2
      orientation_2 = orientation + np.pi/2
      E_max = mu20 * np.sin(orientation_2)**2 - 2*mu11 * np.sin(orientation_2) * np.cos(orientation_2) + mu02 * np.cos(orientation_2)**2
      roundedness = E_min/E_max

      attributes = {
          'position': {'x': float(x_centroid), 'y': float(y_centroid)},
          'orientation': float(orientation),
          'roundness': float(roundedness)
      }

      attribute_list.append(attributes)

  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  print(attribute_list)



if __name__ == '__main__':
  main(sys.argv[1:])

