import cv2
import numpy as np
import glob

def read_images(image_directory):
    # Read all jpg images from the specified directory
    return [cv2.imread(image_path) for image_path in glob.glob(f"{image_directory}/*.jpg")]

def find_image_points(images, pattern_size):
    world_points = []
    image_points = []
    
    # TODO: Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        # Students should fill in code here to generate the world coordinates of the chessboard
        object_point = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
        object_point[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        return object_point
    
    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        # Students should fill in code here to detect corners using cv2.findChessboardCorners or another method
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image, pattern_size, None)
        return corners

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in images:
        corners = detect_corners(image, pattern_size).reshape((-1,2))
        if corners is not None:
            # Add image corners
            image_points.append(corners)
            # Add the corresponding world points
            world_points.append(init_world_points(pattern_size))
    
    return world_points, image_points




def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    
    num_points = len(world_points)
    K = np.zeros((4, 4))
    A = []
    B = []
    P = []

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # The steps are as follows:
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R
    def is_in_center(img_pt, img_size=(640, 480), center_fraction=0.8):
        """检查点是否在图像的中心区域"""
        center_area = [dim * center_fraction for dim in img_size]
        left_bound = (img_size[0] - center_area[0]) / 2
        top_bound = (img_size[1] - center_area[1]) / 2
        right_bound = left_bound + center_area[0]
        bottom_bound = top_bound + center_area[1]

        x, y = img_pt
        return left_bound <= x <= right_bound and top_bound <= y <= bottom_bound

    def centered(world_points, image_points):
        centered_world_points = []
        centered_image_points = []
        for i in range(54):
            if is_in_center(image_points[i]):
                centered_world_points.append(world_points[i])
                centered_image_points.append(image_points[i])
        return np.array(centered_world_points), np.array(centered_image_points)



    # world_points = np.vstack(world_points)
    # image_points = np.vstack(image_points)

    for j in range(13):
        ct_world_points, ct_image_points = centered(world_points[j], image_points[j])
        M= np.zeros((2*ct_world_points.shape[0], 9))
        for i in range(ct_world_points.shape[0]):
            X, Y, Z = ct_world_points[i] 
            u, v = ct_image_points[i]   
            M[i * 2] = [X, Y,  1, 0, 0, 0,  -u * X, -u * Y,  -u]
            M[i * 2 + 1] = [0, 0, 0, X, Y, 1, -v * X, -v * Y,  -v]

        eigenvalues, eigenvectors = np.linalg.eigh(np.dot(M.T, M))
        min_index = np.argmin(eigenvalues)
        H = eigenvectors[:, min_index].reshape((3,3))

        # U,S,Vt=np.linalg.svd(M)
        # H = Vt[-1].reshape(3, 3)

        H = H/H[2,2]
        P.append(H)
        A.extend(
            [
                [H[0, 0] * H[0, 0] - H[0, 1] * H[0, 1],
                 2 * (H[0, 0] * H[1, 0] - H[0, 1] * H[1, 1]), 
                 2 * (H[0, 0] * H[2, 0] - H[0, 1] * H[2, 1]),
                 H[1, 0] * H[1, 0] - H[1, 1] * H[1, 1], 
                 2 * (H[1, 0] * H[2, 0] - H[1, 1] * H[2, 1]),
                 H[2, 0] * H[2, 0] - H[2, 1] * H[2, 1]],

                [
                    H[0,0] * H[0, 1], 
                    H[0, 0] * H[1, 1] + H[0, 1] * H[1, 0], 
                    H[0, 0] * H[2, 1] + H[0, 1] * H[2, 0],
                    H[1, 0] * H[1, 1],
                    H[1, 0] * H[2, 1] + H[1, 1] * H[2, 0], 
                    H[2, 0] * H[2, 1]
                ]
            ]
        )
        
    A = np.array(A)
    eigval, eigvec = np.linalg.eig(A.T@A)
    b = eigvec[:, np.argmin(eigval)]
    b /= np.linalg.norm(b)

    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]],
        
    ])
    
    K = np.linalg.cholesky(B)
    K = np.linalg.inv(K.T)
    K /= K[2,2]
    
    external_matrixs = []
    for i in range(13):  
        E = np.zeros((3, 4))
        R_temp = np.linalg.inv(K)@P[i]
        E[:3, 0] = R_temp[:, 0]
        E[:3, 1] = R_temp[:, 1]
        E[:3, 3] = R_temp[:, 2]
        E[:3, 2] = np.cross(E[:3, 0], E[:3, 1])
        external_matrixs.append(E)

    # Please ensure that the diagonal elements of K are positive
    
    return K, external_matrixs

# Main process
image_path = 'figures'
images = read_images(image_path)

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
pattern_size = (6, 9)  # The pattern size of the chessboard 

np.set_printoptions(precision=2,suppress=True)
world_points, image_points = find_image_points(images, pattern_size)



camera_matrix, camera_extrinsics = calibrate_camera(world_points, image_points)

print("Camera Calibration Matrix:")
print(camera_matrix)



def reprojection_error(world_points, image_points, K, ex_matrixs):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image
    offset_errors = []
    for i in range(len(world_points)):
        # 重投影
        world_points_homogeneous = np.hstack((world_points[i], np.ones((54, 1))))
        camera_points = world_points_homogeneous @ ex_matrixs[i].T
        reprojection_points = camera_points @ K.T
        reprojection_points_normalized = reprojection_points[:, :2] / reprojection_points[:, 2, np.newaxis]
 
        # 计算误差
        error = np.linalg.norm(image_points[i] - reprojection_points_normalized, axis=1)
        offset_errors.append(np.mean(error))

    return offset_errors

offset_errors = reprojection_error(world_points, image_points, camera_matrix, camera_extrinsics)
for i in range(13):
    print("Image ", i,"offset_errors:", offset_errors[i])


def test(image_directory, pattern_size):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded.
    # return None, directly print the results
    # TODO
    images = read_images(image_path)
    world_points, image_points = find_image_points(images, pattern_size)
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, (480, 640), None, None)
    # 构造投影矩阵 P
    P_matrices = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
        P = np.matmul(K, np.hstack((R, tvec)))  # 构造 P = K[R|T]
        P_matrices.append(P)
    print("Camera Matrix:\n", K)


    offset_errors = []
    for i in range(len(world_points)):
        reprojected_points, _ = cv2.projectPoints(world_points[i], rvecs[i], tvecs[i], K, None)
        
        # 计算误差
        error = np.linalg.norm(image_points[i] - reprojected_points.reshape((-1,2)), axis=1)
        # print(error)
        offset_errors.append(np.mean(error))
        print("Image ", i,"offset_errors:", offset_errors[i])


print("Camera Calibration Matrix by OpenCV:")
test(image_path, pattern_size)
