import copy
import os
import shutil
from enum import IntEnum

import cv2 as cv
import skimage as sk
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import natsort as ns
import numpy as np
import sundic.sundic as sdic
import sundic.post_process as sdpp
import sundic.util.datafile as dataFile
from sundic.sundic import CompID, IntConst, ShapeFN
from scipy.interpolate import griddata
from scipy.spatial import KDTree


# Define integer constants for accessing the subset stereo data array
class StereoCompID(IntEnum):
    XCoordID = 0   # The x-coordinate of the subset center point
    YCoordID = 1   # The y-coordinate of the subset center point
    ZCoordID = 2   # The z-coordinate of the subset center point
    XDispID = 3   # The x-displacement of the subset point
    YDispID = 4   # The y-displacement of the subset point
    ZDispID = 5   # The z-displacement of the subset point
    ProjErrID = 6   # The reprojection error of the subset point


# TODO: Check if there are any images
def getStereoImageList(folderPath, debugLevel=0):
    """
    Loads stereo image pairs from either:
    - A folder containing exactly two subfolders (e.g., "0/" or "left" and "1/" or "right"), OR
    - A folder containing all image files suffixed with '_0' for left and '_1' for right images.

    Parameters:
        folderPath (str): Path to the stereo image folder.
        debugLevel (int): Verbosity level (0=silent, 1=summary, 2=detailed).

    Returns:
        tuple: (leftImgSet, rightImgSet)
            leftImgSet (list): List of paths to left camera images.
            rightImgSet (list): List of paths to right camera images.
    """
    if not os.path.isdir(folderPath):
        raise ValueError(f"Folder does not exist: {folderPath}")

    # Store root path and files/folders
    root_path = os.path.abspath(folderPath)
    root_files = os.listdir(root_path)

    # Store supported image file types
    supported_file_types = [".tif", ".tiff", ".png"]

    # Store subfolders if they exist
    subdirs = [dir for dir in root_files if os.path.isdir(os.path.join(root_path, dir))]

    # Case 1: Two subfolders (e.g., "0/", "1/")
    if len(subdirs) == 2:
        # Sort subfolders, assuming first is left and second is right
        subdirs = ns.natsorted(subdirs)
        # Store path to left and right subfolders
        left_path = os.path.join(root_path, subdirs[0])
        right_path = os.path.join(root_path, subdirs[1])
        # Filter, sort and store all files
        left_files = ns.natsorted([
            file for file in os.listdir(left_path)
            if os.path.splitext(file)[1].lower() in supported_file_types
        ])
        right_files = ns.natsorted([
            file for file in os.listdir(right_path)
            if os.path.splitext(file)[1].lower() in supported_file_types
        ])
        # Store lists of full paths to left and right image sets
        leftImgSet = [os.path.join(left_path, file) for file in left_files]
        rightImgSet = [os.path.join(right_path, file) for file in right_files]

    # Case 2: _0 and _1 suffixes
    else:
        # Filter, sort and store all files
        # Left image files end with _0
        left_files = ns.natsorted([
            file for file in os.listdir(root_path)
            if os.path.splitext(file)[1].lower() in supported_file_types and os.path.splitext(file)[0].endswith('_0')
        ])
        # Right image files end with _1
        right_files = ns.natsorted([
            file for file in os.listdir(root_path)
            if os.path.splitext(file)[1].lower() in supported_file_types and os.path.splitext(file)[0].endswith('_1')
        ])

        # Store lists of full paths to left and right image sets
        leftImgSet = [os.path.join(root_path, file) for file in left_files]
        rightImgSet = [os.path.join(root_path, file) for file in right_files]

    # Validate pairing
    if len(leftImgSet) != len(rightImgSet):
        raise ValueError(f"Image pair mismatch: {len(leftImgSet)} left vs {len(rightImgSet)} right")

    if debugLevel > 0:
        print(f"\nLoaded stereo image pairs from: '{folderPath}'")
        print(f"  Left images : {len(leftImgSet)}")
        print(f"  Right images: {len(rightImgSet)}")
        if debugLevel > 1:
            for i, (l, r) in enumerate(zip(leftImgSet, rightImgSet)):
                print(f"  Pair {i+1}:\n    Left : {os.path.basename(l)}\n    Right: {os.path.basename(r)}")

    return leftImgSet, rightImgSet


# TODO: Also return repropjection error
def _triangulatePoints_(leftPts, rightPts, calData):
    """
    Triangulate a 3D points from corresponding set of 2D points.

    This function takes matched 2D points from the left and right stereo images
    and computes their 3D coordinates using the stereo calibration data.

    Parameters:
        - leftPts (ndarray): 2D points from the left image. Shape: (N, 1, 2) or (N, 2)
        - rightPts (ndarray): 2D points from the right image. Shape: (N, 1, 2) or (N, 2)
        - calData (dict): Dictionary containing stereo calibration data.

    Returns:
        - points_3d (ndarray): Array of triangulated 3D points in left
                               camera coordinates. Shape: (N, 3)
    """
    # Extract intrinsic and distortion parameters
    K1, D1 = calData["K1"], calData["D1"]
    K2, D2 = calData["K2"], calData["D2"]
    R, T = calData["R"], calData["T"]

    # Ensure input points are in shape (N, 1, 2)
    # leftPts = np.asarray(leftPts, dtype=np.float32)
    # rightPts = np.asarray(rightPts, dtype=np.float32)

    if leftPts.ndim == 2 and leftPts.shape[1] == 2:
        leftPts = leftPts.reshape(-1, 1, 2)
    if rightPts.ndim == 2 and rightPts.shape[1] == 2:
        rightPts = rightPts.reshape(-1, 1, 2)

    # Undistort the  points using the distortion coefficients
    # pts1 = cv.undistortPoints(leftPts, K1, D1, P=K1)
    # pts2 = cv.undistortPoints(rightPts, K2, D2, P=K2)
    # Undistort into normalised image coordinates
    pts1 = cv.undistortPoints(leftPts, K1, D1)
    pts2 = cv.undistortPoints(rightPts, K2, D2)

    # Reshape to (2, N) for triangulation
    pts1 = pts1.reshape(-1, 2).T
    pts2 = pts2.reshape(-1, 2).T

    # Build projection matrices
    # P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))     # K1 * [I | 0]
    # P2 = K2 @ np.hstack((R, T))                            # K2 * [R | T]
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))     # [I | 0]
    P2 = np.hstack((R, T))                            # [R | T]

    # Triangulate homogeneous 3D points
    points_hom = cv.triangulatePoints(P1, P2, pts1, pts2)  # shape: (4, N)

    # Convert from homogeneous coordinates to 3D: (x, y, z, 1) -> (x, y, z)
    points_3d = cv.convertPointsFromHomogeneous(points_hom.T).reshape(-1, 3)

    # Reproject 3D points onto the left image
    proj_left, _ = cv.projectPoints(points_3d, np.zeros(3), np.zeros(3), K1, D1)

    # Reproject 3D points onto the right image
    rvec, _ = cv.Rodrigues(R)
    proj_right, _ = cv.projectPoints(points_3d, rvec, T, K2, D2)

    # Reshape the points for calculating the reprojection error
    proj_left = proj_left.reshape(-1, 2)
    proj_right = proj_right.reshape(-1, 2)

    leftPts = leftPts.reshape(-1, 2)
    rightPts = rightPts.reshape(-1, 2)

    err_left = np.linalg.norm(leftPts - proj_left, axis=1)
    err_right = np.linalg.norm(rightPts - proj_right, axis=1)

    # Calculate the reprojection error for each point
    reproj_error = np.sqrt((err_left**2 + err_right**2) / 2)

    return points_3d, reproj_error


def _triangulateSubSets_(leftSubSetPnts, rightSubSetPnts, calData):
    """
    Triangulates left and right subset points and returns subset points in world
    coordinates.

    This function extracts the 2D coordinates from the left and right subset
    point arrays, uses the _triangulatePoints_ function to compute their 3D
    positions, and then returns new subset points in the world coordinates.

    Parameters:
        - leftSubSetPnts (ndarray): Array of subset points for the left image.
        - rightSubSetPnts (ndarray): Array of subset points for the right image.
        - calData (dict): Dictionary containing stereo calibration data.

    Returns:
        - worldSubSetPnts (ndarray): Array of subset points in (3D) world coordinates.
    """
    # Extract 2D Points
    # Extract the (x, y) coordinates from the subset point arrays
    left_pts_2d = np.stack((
        leftSubSetPnts[:, :, CompID.XCoordID].flatten(),
        leftSubSetPnts[:, :, CompID.YCoordID].flatten()
    ), axis=1)

    right_pts_2d = np.stack((
        rightSubSetPnts[:, :, CompID.XCoordID].flatten(),
        rightSubSetPnts[:, :, CompID.YCoordID].flatten()
    ), axis=1)

    # Triangulate to get 3D points
    points_3d, reproj_error = _triangulatePoints_(left_pts_2d, right_pts_2d, calData)

    # Create worldSubSetPnts with correct shape to store 3D coordinates and displacements
    subset_grid_shape = (leftSubSetPnts.shape[0], leftSubSetPnts.shape[1], len(StereoCompID))
    worldSubSetPnts = np.zeros(shape=subset_grid_shape)

    # Update the CoordID fields
    data_shape = worldSubSetPnts[:, :, StereoCompID.XCoordID].shape
    worldSubSetPnts[:, :, StereoCompID.XCoordID] = points_3d[:, 0].reshape(data_shape)
    worldSubSetPnts[:, :, StereoCompID.YCoordID] = points_3d[:, 1].reshape(data_shape)
    worldSubSetPnts[:, :, StereoCompID.ZCoordID] = points_3d[:, 2].reshape(data_shape)

    # Update the ProjErrID fields
    worldSubSetPnts[:, :, StereoCompID.ProjErrID] = reproj_error.reshape(data_shape)

    return worldSubSetPnts


def _fillMissingSubsets_(subSetPnts, method='cubic'):
    """
    Fill NaNs in X and Y coordinate fields of subSetPnts using grid interpolation.

    Parameters:
        subSetPnts (ndarray): Subset point array of shape (rows, cols, components).
        method (str): Interpolation method: 'nearest', 'linear', or 'cubic'.

    Returns:
        ndarray: subSetPnts with missing X and Y coordinates filled.
    """
    for coord_id in [CompID.XCoordID, CompID.YCoordID]:
        data = subSetPnts[:, :, coord_id]
        mask = ~np.isnan(data)
        if np.any(~mask):
            rows, cols = np.indices(data.shape)
            known_points = np.stack((rows[mask], cols[mask]), axis=-1)
            known_values = data[mask]
            interp_points = np.stack((rows[~mask], cols[~mask]), axis=-1)
            data[~mask] = griddata(known_points, known_values, interp_points, method=method)
            subSetPnts[:, :, coord_id] = data
    return subSetPnts


def _getEpipolarMask_(Pnts1, Pnts2, calData, Threshold=2.0):
    """
    Computes a mask for corresponding points that fall within a threshold distance
    from their epipolar line constraints.

    Parameters:
        - Pnts1, Pnts2 (Nx2, ndarray), Arrays of corresponding 2D points.
        - calData (dict): Dictionary with stereo calibration data.
        - Threshold (float): Maximum pixel distance from epipolar lines.
                             Defaults to 2.0 pixels.

    """
    # Extract the Fundamental matrix
    # F = calData["F"]

    # Extract the Essential matrix
    E = calData["E"]

    # Extract intrinsic and distortion parameters
    K1, D1 = calData["K1"], calData["D1"]
    K2, D2 = calData["K2"], calData["D2"]

    # Reshape input points to (N, 1, 2)
    Pnts1 = Pnts1.reshape(-1, 1, 2)
    Pnts2 = Pnts2.reshape(-1, 1, 2)

    # Undistort points into normalised image coordinates
    Pnts1_norm = cv.undistortPoints(Pnts1, K1, D1)
    Pnts2_norm = cv.undistortPoints(Pnts2, K2, D2)

    # Reshape points back to (N, 2)
    Pnts1_norm = Pnts1_norm.reshape(-1, 2)
    Pnts2_norm = Pnts2_norm.reshape(-1, 2)

    # Compute the epipolar lines in image 1 for points in image 2
    # Computes ax + by + c = 0 for each point in Nx3 array
    # Lines1 = cv.computeCorrespondEpilines(Pnts2, 2, F)
    Lines1 = cv.computeCorrespondEpilines(Pnts2_norm, 2, E)
    Lines1 = Lines1.reshape(-1, 3)

    # Calculate the distances from Pnts1 to Lines1
    # Distance formula: |ax + by + c | / sqrt(a^2 + b^2)
    num1 = np.abs(Lines1[:,0] * Pnts1_norm[:,0] + Lines1[:,1] * Pnts1_norm[:,1] + Lines1[:,2])
    den1 = np.sqrt(Lines1[:,0]**2 + Lines1[:,1]**2)
    dist1 = num1 / den1

    # Compute the epipolar lines in image 2 for points in image 1
    # Computes ax + by + c = 0 for each point in Nx3 array
    # Lines2 = cv.computeCorrespondEpilines(Pnts1, 1, F)
    Lines2 = cv.computeCorrespondEpilines(Pnts1_norm, 1, E)
    Lines2 = Lines2.reshape(-1, 3)

    # Calculate the distances from Pnts2 to Lines2
    # Distance formula: |ax + by + c | / sqrt(a^2 + b^2)
    num2 = np.abs(Lines2[:,0] * Pnts2_norm[:,0] + Lines2[:,1] * Pnts2_norm[:,1] + Lines2[:,2])
    den2 = np.sqrt(Lines2[:,0]**2 + Lines2[:,1]**2)
    dist2 = num2 / den2

    # Use the maximum of the two as the error
    error = np.maximum(dist1, dist2)

    # Average focal length of both cameras
    f_avg = (K1[0,0] + K1[1,1] + K2[0,0] + K2[1,1]) / 4.0

    # Calculate normalised threshold, as input threshold is in pixel coordinates
    Threshold_norm = Threshold / f_avg

    # Create boolean mask based on threshold
    mask = (error < Threshold_norm).astype(np.uint8)

    return mask


# TODO: doc string
# TODO: Why is 8 selected as the minimum keypoints here?
def _getStereoKeyPoints_(leftImg, rightImg, ROI):
    # Read and normalize the images to be in the range 0-255
    left_img_gray = sdic.readImage(leftImg, normalize8Bit=True)
    right_img_gray = sdic.readImage(rightImg, normalize8Bit=True)

    x_start, y_start, w, h = ROI
    left_mask = np.zeros_like(left_img_gray, dtype=np.uint8)
    left_mask[y_start:y_start+h, x_start:x_start+w] = 255

    # TODO: Maybe increased threshold to decrease amount of points for better performance?
    # akaze = cv.xfeatures2d.AKAZE_create(threshold=0.003)
    akaze = cv.xfeatures2d.AKAZE_create()
    # akaze = cv.SIFT_create()    # Experiment with SIFT
    kp1, desc1 = akaze.detectAndCompute(left_img_gray, mask=left_mask)
    kp2, desc2 = akaze.detectAndCompute(right_img_gray, mask=None)

    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("AKAZE: Not enough feature points found for matching.")

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # For AKAZE
    # matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)  # For SIFT
    matches = matcher.match(desc1, desc2)

    if len(matches) < 8:
        raise RuntimeError("AKAZE: Not enough valid matches found.")

    # DEBUG: Amount of keypoints filtered by BF Matcher
    print(f"Keypoints removed by BF Matcher: {len(kp1) - len(matches)}")

    akaze_pts_left = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    akaze_pts_right = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # DEBUG: Number of keypoints found
    print(f"Found {len(akaze_pts_left)} keypoints.")

    return akaze_pts_left, akaze_pts_right


def _filterStereoKeyPoints_(akaze_pts_left, akaze_pts_right, calData):
    if calData is None:
        raise ValueError("Calibration data ('calData') is incorrect or missing.")

    # Epipolar filtering
    mask_epipolar = _getEpipolarMask_(akaze_pts_left, akaze_pts_right, calData, Threshold=50.0)
    if mask_epipolar is None or np.sum(mask_epipolar) == 0:
        raise RuntimeError("AKAZE: Epipolar filtering failed.")

    # DEBUG: Amount of keypoints filtered by epipolar constraint
    print(f"Keypoints removed by Epipolar Constraint: {len(akaze_pts_left) - np.sum(mask_epipolar)}")

    valid_epi = (mask_epipolar.ravel() == 1)
    akaze_pts_left = akaze_pts_left[valid_epi]
    akaze_pts_right = akaze_pts_right[valid_epi]

    return akaze_pts_left, akaze_pts_right


# TODO: Rewrite to use multiple cores, as each subset is independent.
# TODO: Maybe vectorise it and use workers=N_CORES
def _estimateSubSetsAkaze_(subSetPnts_left, subSetPnts_right, akaze_left, akaze_right, N_neighbors, search_radius):
    """
    Estimate the first order shape function parameters for the right image subsets using
    AKAZE keypoints.

    Fallback strategy:
    1. Use the N neighbours if they are within the search radius (defaults to subset size).
    2. Else use the N nearest neighbours if the total number of keypoints >= N.
    3. Else use a least squares fit over the keypoints.

    Parameters:
        subSetPnts_left (ndarray): The reference left image subset array.
        subSetPnts_right (ndarray): The initial right image subset array. Copy of subSetPnts_left if None.
        akaze_left (ndarray): The array of left image AKAZE keypoints.
        akaze_right (ndarray):: The array of left image AKAZE keypoints.
        N_neighbors: Number of neighbouring keypoints to use for RANSAC model fitting.
        search_radius: The pixel radius around the subset centre used to search for keypoints.
    Returns:
        subSetPnts_right (ndarray): Array of subset points with estimated shape function
                                    parameters for the right image.

    Raises:
        RuntimeError: If not enough keypoints are provided.
    """
    if len(akaze_left) < 7:
        raise RuntimeError("RANSAC: At least 7 keypoints are required.")

    # Create initial right image subsets by copying left image subsets if none are given.
    if subSetPnts_right is None:
        subSetPnts_right = np.copy(subSetPnts_left)

    # Create KDTree for left keypoints
    tree = KDTree(akaze_left)
    # Number of subset rows and columns
    rows, cols = subSetPnts_left.shape[:2]

    # For each left image subset
    for iRow in range(rows):
        for iCol in range(cols):
            # Get left image subset x and y coordinates
            x = subSetPnts_left[iRow, iCol, CompID.XCoordID]
            y = subSetPnts_left[iRow, iCol, CompID.YCoordID]

            # Ignore if subset is invalid
            if np.isnan(x) or np.isnan(y):
                continue

            # Find N keypoints within the search radius
            idxs = tree.query_ball_point([x, y], search_radius)
            idxs = np.atleast_1d(idxs)

            # If there are not enough keypoints within the search radius
            if len(idxs) < N_neighbors:
                # print(f"DEBUG: Fell through to KNN at [{x}, {y}]")
                # Find N nearest keypoints to subset x,y position
                dists, idxs = tree.query([x, y], k=N_neighbors)
                idxs = np.atleast_1d(idxs)

            # Extract keypoints that will be used to fit the RANSAC model
            src_pts = akaze_left[idxs]
            dst_pts = akaze_right[idxs]

            # Perform RANSAC model fitting to estimate affine parameters
            # RANSAC requires at least 7 samples
            if len(idxs) >= N_neighbors:
                model, inliers = sk.measure.ransac(
                    (src_pts, dst_pts),
                    sk.transform.AffineTransform,
                    min_samples=3,  # Requires at least 3 samples to be solvable
                    residual_threshold=2.0,
                    max_trials=100
                )
            else:
                print(f"DEBUG: Fell through to LSM at [{x}, {y}]")
                # If there are not enough keypoints, perform a least squares fit
                model = sk.transform.AffineTransform()
                model.from_estimate(dst_pts)  # Not sure if this is a good idea?

            # Apply estimated affine parameters to right image subsets
            if model is not None and model.params is not None:
                # | x'|   | a0 a1 a2 | | x |
                # | y'| = | b0 b1 b2 | | y |
                # | 1 |   | 0  0  1  | | 1 |
                # x' = a0 * x + a1 * y + a2
                x_prime = model.params[0, 0] * x + model.params[0, 1] * y + model.params[0, 2]
                # y' = b0 * x + b1 * y + b2
                y_prime = model.params[1, 0] * x + model.params[1, 1] * y + model.params[1, 2]
                #            | 1+p2  p3   p1 | | x |
                # W(x,y,p) = |  p5  1+p6  p4 | | y |
                #            |  0    0    1  | | 1 |
                subSetPnts_right[iRow, iCol, CompID.XDispID + 0] = x_prime - x               # p1
                subSetPnts_right[iRow, iCol, CompID.XDispID + 1] = model.params[0, 0] - 1.0  # p2
                subSetPnts_right[iRow, iCol, CompID.XDispID + 2] = model.params[0, 1]        # p3

                subSetPnts_right[iRow, iCol, CompID.YDispID + 0] = y_prime - y               # p4
                subSetPnts_right[iRow, iCol, CompID.YDispID + 1] = model.params[1, 0]        # p5
                subSetPnts_right[iRow, iCol, CompID.YDispID + 2] = model.params[1, 1] - 1.0  # p6

    return subSetPnts_right


# TODO: Rewrite so that it supports multiple strategies
# TODO: Rewrite so that it will perform stereo matching using existing subsets
#       i.e.: stereo match each stereo image pair
def stereoMatch(settings, calData, leftImg, rightImg, fillMissing=False):
    """
    Perform stereo matching between the leftImg (reference) and rightImg
    (target), and return the leftSubSetPnts for the leftImg and the matched
    corresponding rightSubSetPnts for the rightImg.

    Parameters:
        settings (Settings): A Settings object containing the settings for the DIC analysis.
        leftImg (str): Path to the left stereo image (reference).
        rightImg (str): Path to the right stereo image.
        calData (dict): Dictionary with stereo calibration data.
        fillMissing (bool): Fill in missing subsets on the right image via
                            interpolation. Defaults to False.

    Returns:
        tuple:
            leftSubSetPnts (ndarray): Array of subset points for the left image.
            rightSubSetPnts (ndarray): Array of corresponding subset points for the right image.

    Raises:
        FileNotFoundError: If the specified image files cannot be loaded.
    """
    # 1. Setup
    # Create copy of user settings for stereo matching specific modification
    sm_settings = copy.deepcopy(settings)

    # Create a temporary folder for intermediate files
    tempFolderPath = os.path.join(os.getcwd(), "sm_temp")
    if os.path.exists(tempFolderPath):
        shutil.rmtree(tempFolderPath)

    os.makedirs(tempFolderPath, exist_ok=True)

    # Check if left and right images exist
    if not os.path.exists(leftImg) or not os.path.exists(rightImg):
        raise FileNotFoundError(f"Could not load images: {leftImg}, {rightImg}")

    # Copy images to tempFolderPath
    shutil.copyfile(leftImg, os.path.join(tempFolderPath, "sm_img_0.tif"))
    shutil.copyfile(rightImg, os.path.join(tempFolderPath, "sm_img_1.tif"))

    # # Just for testing
    # histMatch = True
    # if histMatch:
    #     left_img_to_process = cv.imread(leftImg, cv.IMREAD_GRAYSCALE)
    #     right_img_to_process = cv.imread(rightImg, cv.IMREAD_GRAYSCALE)
    #     right_img_to_process = match_histograms(right_img_to_process, left_img_to_process)
    #     right_img_to_process = np.clip(right_img_to_process, 0, 255).astype(np.uint8)
    #     cv.imwrite(os.path.join(tempFolderPath, "sm_img_0.tif"), left_img_to_process)
    #     cv.imwrite(os.path.join(tempFolderPath, "sm_img_1.tif"), right_img_to_process)
    #     if sm_settings.DebugLevel >= 1:
    #         print("Applied histogram matching to right image.")

    # 2. DIC Analysis
    sm_settings.ImageFolder = tempFolderPath
    # sm_settings.ShapeFunctions = "Quadratic"
    # sm_settings.CPUCount = 1       # TODO: Multiproccessing currently broken for stereo matching
    sm_settings.ReferenceStrategy = "Absolute"
    sm_settings.DatumImage = 0
    sm_settings.TargetImage = -1
    sm_settings.Increment = 1
    sm_settings.StartingPoints = 0  # Disable initial shape function parameter estimation
    resultsPath = os.path.join(tempFolderPath, "sm_results.sdic")

    # Get the Region of Interest (ROI) for the left image
    ROI = sdic._setupROI_(sm_settings.ROI, [leftImg, rightImg], debugLevel=sm_settings.DebugLevel)

    # Create the initial subsets for the left image
    subSetSize = sm_settings.SubsetSize
    stepSize = sm_settings.StepSize
    shapeFn = sm_settings.ShapeFunctions
    initSubSetPnts_left = sdic._setupSubSets_(
        subSetSize, stepSize, shapeFn, ROI, leftImg, debugLevel=sm_settings.DebugLevel)
    if initSubSetPnts_left.size == 0:
        raise ValueError("No valid subset centers could be created for the specified ROI/subset size.")

    # DEBUG: Plot initial left subsets on image
    plotSubSetsOnImage(leftImg,
                       initSubSetPnts_left,
                       applyDeformation=True,
                       showCenters=True,
                       maxSubSets=None,
                       title="Initial left image subsets",
                       fileName="initSubSetPnts_left.png",
                       showPlot=True)

    # Get AKAZE keypoints
    print("Get AKAZE keypoints")
    sdic._tic_()
    # Elapsed time is 5.5641326904296875 seconds.
    akaze_left, akaze_right = _getStereoKeyPoints_(leftImg, rightImg, ROI)
    sdic._toc_()

    # TODO: Implement keypoint subsampling

    # Filter keypoints
    print("Filter keypoints")
    sdic._tic_()
    # Elapsed time is 0.0025370121002197266 seconds.
    akaze_left, akaze_right = _filterStereoKeyPoints_(akaze_left, akaze_right, calData)
    sdic._toc_()

    # DEBUG: Plot keypoints in 3D
    # plotPoints3D(points_3d, fileName="keypoints_3d.png", showPlot=True, set_aspect="equal")

    # Estimate right image subset parameters using keypoints
    print("Estimate right image subset parameters")
    N_neighbors = 14
    search_radius = subSetSize / np.sqrt(2)  # Search circle around subset
    sdic._tic_()
    initSubSetPnts_right = _estimateSubSetsAkaze_(initSubSetPnts_left, None, akaze_left, akaze_right, N_neighbors, search_radius)
    sdic._toc_()

    # Create and export 3D mesh
    # _ = _createMesh3D_(points_3d, export=True)

    # DEBUG: Plot initial right image subset points before matching
    plotSubSetsOnImage(rightImg,
                       initSubSetPnts_right,
                       applyDeformation=True,
                       showCenters=True,
                       maxSubSets=None,
                       title="Initial right image subset estimates",
                       fileName="initSubSetPnts_right.png",
                       showPlot=True)
    # breakpoint()

    # Perform temporal matching to get the right image subset coordinates and
    # displacements
    returnData = sdic._temporalMatch_(initSubSetPnts_right, [leftImg, rightImg], sm_settings, resultsPath)
    rightSubSetPnts = np.copy(returnData[0])
    leftSubSetPnts = np.copy(returnData[0])

    # Optional debug output for subset matching statistics
    if sm_settings.DebugLevel >= 2:
        results, nRows, nCols = sdpp.getDisplacements(resultsPath, -1)
        results = results[~np.isnan(results).any(axis=1)]
        foundPoints = results.shape[0]
        totalPoints = nRows * nCols
        missingPoints = totalPoints - foundPoints
        print(f"Found {foundPoints}/{totalPoints} subsets. Missing: {missingPoints}")

    # 3. Post-processing
    # Update right subset coordinates with calculated displacements
    rightSubSetPnts[:, :, CompID.XCoordID] += rightSubSetPnts[:, :, CompID.XDispID]
    rightSubSetPnts[:, :, CompID.YCoordID] += rightSubSetPnts[:, :, CompID.YDispID]

    # Zero out displacements and reset the correlation metric
    rightSubSetPnts[:, :, CompID.XDispID:] = 0.0
    rightSubSetPnts[:, :, CompID.CZNSSDID] = IntConst.CNZSSD_MAX
    leftSubSetPnts[:, :, CompID.XDispID:] = 0.0
    leftSubSetPnts[:, :, CompID.CZNSSDID] = IntConst.CNZSSD_MAX

    # Optionally fill missing subsets
    # TODO: Make filling more robust
    # TODO: Implement OpenCorr regional fitting
    if fillMissing and missingPoints:
        rightSubSetPnts = _fillMissingSubsets_(rightSubSetPnts)
        if sm_settings.DebugLevel >= 1:
            print("Filled in missing subsets for right image.")

    # DEBUG: Plot right image subsets on image after stereo matching
    plotSubSetsOnImage(rightImg,
                       rightSubSetPnts,
                       applyDeformation=True,
                       showCenters=True,
                       maxSubSets=None,
                       title="Initial right image subsets after matching",
                       fileName="matchedSubSetPnts_right.png",
                       showPlot=True)

    # 4. Cleanup
    if sm_settings.DebugLevel < 2:
        shutil.rmtree(tempFolderPath)

    return leftSubSetPnts, rightSubSetPnts


# TODO: Add option to set calibration parameters location in settings file. The
# parameters should be stored in a csv file in the required format.
# For now, first get the calibration data outside the function by performing
# stereo calibration or read it from a csv file using sc.getDataFromParametersCSV(calCSV)
# TODO: Add option in settings to fillMissing subset points
# TODO: Use ray to run temporalMatch in parallel
# TODO: Implement and use absolute DIC strategy
# TODO: Rewrite the function to make use of files instead of memory
def stereoDICLocal(settings, calData, resultsFile, fillMissing=False):

    # 1. Get left and right image sets
    leftImgSet, rightImgSet = getStereoImageList(settings.ImageFolder, debugLevel=settings.DebugLevel)

    # 2. Check if calData is present
    if calData is None:
        raise ValueError("Calibration data ('calData') is incorrect or missing.")

    # 3. Perform stereo matching on the first image pair to create the initial
    #    leftSubSetPnts and matching rightSubSetPnts
    refLeftImg = leftImgSet[settings.DatumImage]
    refRightImg = rightImgSet[settings.DatumImage]
    leftSubSetPnts_init, rightSubSetPnts_init = stereoMatch(settings, calData, refLeftImg, refRightImg, fillMissing)

    # 4. Perform temporal matching on the left image set
    if settings.DebugLevel > 0:
        print("Performing temporal matching on the left image set...")
    results_left = sdic._temporalMatch_(leftSubSetPnts_init, leftImgSet, settings, "tm_results_left.sdic")

    # 5. Perform temporal matching on the right image set
    if settings.DebugLevel > 0:
        print("Performing temporal matching on the right image set...")
    results_right = sdic._temporalMatch_(rightSubSetPnts_init, rightImgSet, settings, "tm_results_right.sdic")

    # DEBUG: Plot final left image SubSetPnts
    finalLeftImg = leftImgSet[settings.TargetImage]
    plotSubSetsOnImage(finalLeftImg,
                       results_left[-1],
                       applyDeformation=True,
                       showCenters=True,
                       maxSubSets=None,
                       title="Final left image subsets",
                       fileName="finalSubSetPnts_left.png",
                       showPlot=True)

    # DEBUG: Plot final right image SubSetPnts
    finalRightImg = rightImgSet[settings.TargetImage]
    plotSubSetsOnImage(finalRightImg,
                       results_right[-1],
                       applyDeformation=True,
                       showCenters=True,
                       maxSubSets=None,
                       title="Final right image subsets",
                       fileName="finalSubSetPnts_right.png",
                       showPlot=True)

    # 6. Perform coordinate system transform to transform the displacements from
    #    the left image plane and right image plane (2D) to the world
    #    coordinates (3D)

    # Get the temporal image pair information
    imgDatum = settings.DatumImage
    imgTarget = settings.TargetImage
    if imgTarget == -1:
        imgTarget = len(leftImgSet) - 1
    imgIncr = settings.Increment
    # imgPairs = int((imgTarget - imgDatum) / imgIncr)

    # TODO: Create new data structure for world results

    # Prepare results file writer for stereo results
    df = dataFile.DataFile.openWriter(resultsFile)
    df.writeHeading(settings)

    returnData = []
    # Initial triangulation, always from imgDatum
    worldSubSetPnts_ref = _triangulateSubSets_(leftSubSetPnts_init, rightSubSetPnts_init, calData)
    worldSubSetPnts_out = np.copy(worldSubSetPnts_ref)

    # For each stereo image pair, perform least-squares triangulation on the
    # left and right subset points to get the world subset points
    for imgPairIdx, img in enumerate(range(imgDatum, imgTarget, imgIncr)):
        print(f"imgPairIdx: {imgPairIdx}, img: {img}")

        # Get left and right image subsets
        leftSubSetPnts = results_left[imgPairIdx]
        rightSubSetPnts = results_right[imgPairIdx]

        # Update left and right subset coordinates with calculated displacements
        # to calculate 3D displacements from them
        rightSubSetPnts[:, :, CompID.XCoordID] += rightSubSetPnts[:, :, CompID.XDispID]
        rightSubSetPnts[:, :, CompID.YCoordID] += rightSubSetPnts[:, :, CompID.YDispID]
        leftSubSetPnts[:, :, CompID.XCoordID] += leftSubSetPnts[:, :, CompID.XDispID]
        leftSubSetPnts[:, :, CompID.YCoordID] += leftSubSetPnts[:, :, CompID.YDispID]

        # Assume relative strategy for now, but when using absolute, the
        # _triangulateSubSets_ function should not update the world subset
        # coordinates, only the displacements. Therefore I will have to create a
        # new function that performs triangulation to get the new subset world
        # coordinates, then only update the displacements of worldSubSetPnts
        # using the newly calculated coordinates, leaving the coordinates of
        # worldSubSetPnts as is.

        # The _triangulateSubSets_ function returns the a subSetPnts array
        # containing the newly triangulated x,y and z coordinates, but doesn't
        # touch displacement

        # Triangulate current stereo pair
        worldSubSetPnts = _triangulateSubSets_(leftSubSetPnts, rightSubSetPnts, calData)

        # Calculate displacements from the reference
        worldSubSetPnts_out[:, :, StereoCompID.XDispID] = worldSubSetPnts[:, :, StereoCompID.XCoordID] - worldSubSetPnts_ref[:, :, StereoCompID.XCoordID]
        worldSubSetPnts_out[:, :, StereoCompID.YDispID] = worldSubSetPnts[:, :, StereoCompID.YCoordID] - worldSubSetPnts_ref[:, :, StereoCompID.YCoordID]
        worldSubSetPnts_out[:, :, StereoCompID.ZDispID] = worldSubSetPnts[:, :, StereoCompID.ZCoordID] - worldSubSetPnts_ref[:, :, StereoCompID.ZCoordID]

        # TODO: Figure out if the subset coordinates are ever updated in SUN-DIC, as it doesn't seem so
        # Update coordinates from triangulation
        # worldSubSetPnts_out[:, :, CompID.XCoordID] = worldSubSetPnts[:, :, CompID.XCoordID]
        # worldSubSetPnts_out[:, :, CompID.YCoordID] = worldSubSetPnts[:, :, CompID.YCoordID]
        # worldSubSetPnts_out[:, :, CompID.ZCoordID] = worldSubSetPnts[:, :, CompID.ZCoordID]

        # TODO: Ask Prof. Venter why the subset coordinates are always set to
        # their initial positions?

        # Write this frame's results to the results file
        df.writeSubSetData(imgPairIdx, worldSubSetPnts_out)

        returnData.append(worldSubSetPnts_out)

    df.close()

    return returnData


# The following function has been generated using Gemini.
# TODO: Implement into sundic postprocessing
def plotSubSetsOnImage(img, subSetPnts, applyDeformation=True, showCenters=True, maxSubSets=None, title="DIC Subsets on Image", fileName="subsets.png", showPlot=True):
    """
    Plots the subsets on the provided image with their deformed shapes.

    Parameters:
        - img: Image file path (str) or loaded image array (ndarray).
        - subSetPnts (ndarray): The subset points array.
        - applyDeformation (bool): Whether to plot subsets with their deformed shapes.
        - showCenters (bool): Whether to plot a marker at the center of each subset.
        - maxSubSets (int, optional): Maximum number of subsets to plot to avoid clutter.
        - fileName (str): Path to save the plot image.
        - showPlot (bool): Whether to display the plot interactively.
    """
    # Read the image if a file path is provided
    if isinstance(img, str):
        image = cv.imread(img, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image from '{img}'")
    else:
        image = img

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')

    # Flatten the 3D subset points array to a 2D array for easier iteration
    flat_subsets = subSetPnts.reshape(-1, subSetPnts.shape[-1])

    # Filter for active and valid subsets
    valid_mask = ~np.isnan(flat_subsets[:, CompID.XCoordID]) & \
        ~np.isnan(flat_subsets[:, CompID.YCoordID])

    valid_subsets = flat_subsets[valid_mask]

    # Subsample if maxSubSets is specified and exceeded
    if maxSubSets is not None and maxSubSets > 0 and len(valid_subsets) > maxSubSets:
        indices = np.linspace(0, len(valid_subsets) - 1, maxSubSets, dtype=int)
        valid_subsets = valid_subsets[indices]

    for subset in valid_subsets:
        x0 = subset[CompID.XCoordID]
        y0 = subset[CompID.YCoordID]
        ss = subset[CompID.SSSizeID]
        shape_fn = int(subset[CompID.ShapeFnID])

        hw = (ss - 1) / 2.0

        # Create points along the perimeter of the undeformed square
        # Using multiple points per edge ensures smooth boundaries for quadratic deformation
        edge_points = 10

        # Top edge
        xsi_top = np.linspace(-hw, hw, edge_points)
        eta_top = np.full_like(xsi_top, -hw)
        # Right edge
        eta_right = np.linspace(-hw, hw, edge_points)
        xsi_right = np.full_like(eta_right, hw)
        # Bottom edge
        xsi_bottom = np.linspace(hw, -hw, edge_points)
        eta_bottom = np.full_like(xsi_bottom, hw)
        # Left edge
        eta_left = np.linspace(hw, -hw, edge_points)
        xsi_left = np.full_like(eta_left, -hw)

        xsi = np.concatenate([xsi_top, xsi_right, xsi_bottom, xsi_left])
        eta = np.concatenate([eta_top, eta_right, eta_bottom, eta_left])

        if applyDeformation:
            # Model coefficients span from XDispID to XDispID + 12
            p = subset[CompID.XDispID : CompID.XDispID + 12]
        else:
            p = np.zeros(12)

        # Apply displacement mapping
        if shape_fn == ShapeFN.AFFINE:
            xsi_d = (1 + p[1]) * xsi + p[2] * eta + p[0]
            eta_d = p[7] * xsi + (1 + p[8]) * eta + p[6]
        elif shape_fn == ShapeFN.QUADRATIC:
            xsi_d = 0.5 * p[3] * (xsi**2) + p[4] * (xsi * eta) + 0.5 * p[5] * (eta**2) + \
                (1 + p[1]) * xsi + p[2] * eta + p[0]
            eta_d = 0.5 * p[9] * (xsi**2) + p[10] * (xsi * eta) + 0.5 * p[11] * (eta**2) + \
                p[7] * xsi + (1 + p[8]) * eta + p[6]
        else:
            xsi_d = xsi
            eta_d = eta

        # Compute absolute deformed coordinates
        x_d = x0 + xsi_d
        y_d = y0 + eta_d

        # Plot polygon outline
        poly_points = np.column_stack((x_d, y_d))
        polygon = Polygon(poly_points, closed=True, edgecolor='red', facecolor='none', linewidth=1.5, alpha=0.8)
        ax.add_patch(polygon)

        # Plot deformed center point
        if showCenters:
            cx = x0 + p[0]
            cy = y0 + p[6]
            ax.plot(cx, cy, marker='+', color='blue', markersize=5)

    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_aspect('equal')

    plt.tight_layout()

    if fileName:
        plt.savefig(fileName, dpi=300)
        print(f"Plot saved to '{fileName}'")

    if showPlot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plotStereoSubSets2D(leftSubSetPnts, rightSubSetPnts, fileName="subset_points_2d.png", showPlot=False):
    """
    Plot both the leftSubSetPnts and rightSubSetPnts on a 2D scatter plot.

    Parameters:
        - leftSubSetPnts (ndarray): Array of subset points for the left image.
        - rightSubSetPnts (ndarray): Array of subset points for the right image.
        - fileName (str): File path to where the plot will be saved.
                          Default is 'subset_points_2d.png'
        - showPlot (bool): Display the plot window. Default is True.
    """
    # Extract subset points
    left_pts = np.stack((
        leftSubSetPnts[:, :, CompID.XCoordID].flatten(),
        leftSubSetPnts[:, :, CompID.YCoordID].flatten()
    ), axis=1)

    right_pts = np.stack((
        rightSubSetPnts[:, :, CompID.XCoordID].flatten(),
        rightSubSetPnts[:, :, CompID.YCoordID].flatten()
    ), axis=1)

    # Plot subsets
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(*left_pts.T, s=5, label="Left", color="blue")
    plt.scatter(*right_pts.T, s=5, label="Right", color="red")
    plt.axis("equal")
    plt.title("Left and Right Image Subset Points")
    plt.legend()

    # Save plot
    plt.savefig(fileName)
    print(f"Plot saved to '{fileName}'")

    # Show plot window
    if showPlot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plotSubSets3D(worldSubSetPnts, fileName="subset_points_3d.png", showPlot=False, set_aspect="auto"):
    """
    Plot the worldSubSetPnts on a 3D scatter plot.

    Left camera focal point used as origin in world coordinates.

    Parameters:
        - worldSubSetPnts (ndarray): Array of subset points in world (3D) coordinates.
        - fileName (str): Path where the plot will be saved.
        - showPlot (bool): Whether to display the plot window. Defaults to False
        - set_aspect (str): Set plot axis scaling, can be one of the following
                            {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
                            Defaults to 'auto'
    """
    # Extract 3D coordinates
    pts = np.stack((
        worldSubSetPnts[:, :, StereoCompID.XCoordID].flatten(),
        worldSubSetPnts[:, :, StereoCompID.YCoordID].flatten(),
        worldSubSetPnts[:, :, StereoCompID.ZCoordID].flatten()
    ), axis=1)

    # Remove invalid points (NaN or Inf)
    valid_mask = np.isfinite(pts).all(axis=1)
    pts = pts[valid_mask]

    if pts.size == 0:
        print("No valid 3D points to plot.")
        return

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*pts.T, s=5, color="blue")

    # Axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Left Subset Points in 3D")
    ax.grid(True)

    # Set axis scaling
    ax.set_aspect(set_aspect)

    plt.tight_layout()

    # Save or display
    plt.savefig(fileName)
    print(f"Plot saved to '{fileName}'")
    if showPlot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plotDispContour3D(worldSubSetPnts,
                      dispComp="z",
                      fileName="displacement_contour_3d.png",
                      showPlot=False,
                      set_aspect="auto"):
    """
    Create a 3D scatter plot where the color represents displacement in the selected direction
    or total displacement magnitude.

    Parameters:
        - worldSubSetPnts (ndarray): 3D array with point coordinates and displacements.
        - dispComp (str): Which displacement to use for coloring ("x", "y", "z", or "mag").
        - fileName (str): Path to save the plot image.
        - showPlot (bool): Whether to display the plot interactively.
        - set_aspect (str): Axis scaling. {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
    """
    # Map component selection to index
    disp_map = {
        "x": StereoCompID.XDispID,
        "y": StereoCompID.YDispID,
        "z": StereoCompID.ZDispID,
    }

    # Extract 3D coordinates
    x = worldSubSetPnts[:, :, StereoCompID.XCoordID].flatten()
    y = worldSubSetPnts[:, :, StereoCompID.YCoordID].flatten()
    z = worldSubSetPnts[:, :, StereoCompID.ZCoordID].flatten()

    # Get displacements
    dx = worldSubSetPnts[:, :, StereoCompID.XDispID].flatten()
    dy = worldSubSetPnts[:, :, StereoCompID.YDispID].flatten()
    dz = worldSubSetPnts[:, :, StereoCompID.ZDispID].flatten()

    # Determine displacement values for coloring
    if dispComp == "mag":
        disp = np.sqrt(dx**2 + dy**2 + dz**2)
        colorbar_label = "Total Displacement Magnitude"
        title = "3D Displacement Contour - Magnitude"
    elif dispComp in disp_map:
        disp = worldSubSetPnts[:, :, disp_map[dispComp]].flatten()
        colorbar_label = f"{dispComp.upper()} Displacement"
        title = f"3D Displacement Contour - {dispComp.upper()} Direction"
    else:
        raise ValueError("Invalid displacement component. Choose from 'x', 'y', 'z', or 'mag'.")

    # Remove invalid points
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(disp)
    x, y, z, disp = x[valid_mask], y[valid_mask], z[valid_mask], disp[valid_mask]

    if x.size == 0:
        print("No valid 3D points to plot.")
        return

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use displacement as color
    scatter = ax.scatter(x, y, z, c=disp, cmap='viridis', s=8, alpha=0.9)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(colorbar_label)

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set axis scaling
    ax.set_aspect(set_aspect)

    plt.tight_layout()

    # Save or show
    plt.savefig(fileName)
    print(f"Displacement contour plot saved to '{fileName}'")
    if showPlot:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plotPoints3D(points_3d, fileName="points_3d.png", showPlot=False, set_aspect="auto"):
    """
    Plot 3D triangulated keypoints on a scatter plot.

    Left camera focal point is used as the origin (0, 0, 0) in world coordinates.

    Parameters:
        - points_3d (ndarray): Array of 3D points. Expected shape (N, 3).
        - fileName (str): Path where the plot image will be saved.
        - showPlot (bool): Whether to display the interactive plot window. Defaults to False.
        - set_aspect (str): Set plot axis scaling ('auto', 'equal', etc.). Defaults to 'auto'.
    """
    if points_3d is None or len(points_3d) == 0:
        print("No valid 3D points to plot.")
        return

    # Ensure shape is (N, 3)
    pts = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)

    # Filter out invalid points (NaN or Inf)
    valid_mask = np.isfinite(pts).all(axis=1)
    pts = pts[valid_mask]

    if pts.size == 0:
        print("No valid 3D points remaining after NaN/Inf filtering.")
        return

    # Create 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points colored by Z depth
    scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis', s=10, alpha=0.8)
    fig.colorbar(scatter, ax=ax, label='Z Depth (mm)', shrink=0.6)

    # Axis labels and title
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Triangulated Keypoints in 3D (N={len(pts)})")
    ax.grid(True)

    # Handle aspect ratio scaling
    if set_aspect == 'equal':
        max_range = np.array([
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            pts[:, 2].max() - pts[:, 2].min()
        ]).max() / 2.0

        mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
        mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
        mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        try:
            ax.set_aspect(set_aspect)
        except Exception:
            ax.set_aspect('auto')

    plt.tight_layout()

    # Save or Display
    plt.savefig(fileName, dpi=300)
    print(f"Plot saved to '{fileName}'")

    if showPlot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
