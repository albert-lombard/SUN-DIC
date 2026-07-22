################################################################################
# Stereo Camera Calibration Module
#
# This file contains functions for performing stereo camera calibration.
# It processes images of a checkerboard calibration target to compute the
# intrinsic parameters of each camera, as well as the extrinsic parameters
# that describe their relative position and orientation.
#
# Author: H.A.J. Lombard
# Date: 2026/07/21
################################################################################

# Example usage:
"""
import stereo_calibrate as sc
import glob

# Set calibration target parameters
board_size = (7, 10)            # (row, column), number of inner corners per row and column
square_size = 10                # Square size in [mm]

# Collect images
left_images = glob.glob('../DuoDIC_sample_data/calibration_11_18_10mm/cam_01/*.tiff')
right_images = glob.glob('../DuoDIC_sample_data/calibration_11_18_10mm/cam_02/*.tiff')

# Perform camera stereo calibration and get calibration data
caldata = sc.calibrate_checkerboard(board_size, square_size, left_images, right_images)

# Or load calibration data from file
with open("../data/chb_caldata.pkl", "rb") as f:
    caldata = pickle.load(f)
"""

# References:
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://docs.python.org/3/library/concurrent.futures.html

# Import built-in libraries
import csv
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Import external libraries
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# ====================
#      Functions
# ====================

# The problem with the following function is that it sets the reference
# orientation of the grid to that of the first image, even if the first image
# has an orientation that differs with the rest of the images. Works ok for now.
def _isOrientationConsistent_(corners1, corners2, referenceDirection):
    """
    Check if the orientation of the checkerboard grid is consistent across images.

    Parameters:
        - corners1 (ndarray): Coordinates of the detected corners for the left image.
        - corners2 (ndarray): Coordinates of the detected corners for the right image.
        - referenceDirection (list): A list containing a single numpy array reference
                                     orientation vector.

    Returns:
        - bool: True if orientations are consistent, False otherwise.
    """
    # Compute orientation vectors (last - second-last corner)
    direction1 = corners1[-1][0] - corners1[-2][0]
    direction2 = corners2[-1][0] - corners2[-2][0]

    # Normalize vectors
    direction1 /= np.linalg.norm(direction1)
    direction2 /= np.linalg.norm(direction2)

    # Set reference direction if not already set
    if referenceDirection[0] is None:
        referenceDirection[0] = (direction1 + direction2) / 2 # Average
        referenceDirection[0] /= np.linalg.norm(referenceDirection[0])  # Normalize
        # print("Set reference orientation.")
        return True

    # Check if orientations align with the reference
    dot1 = np.dot(referenceDirection[0], direction1)
    dot2 = np.dot(referenceDirection[0], direction2)

    # Check if orientations align between the two images
    pair_dot = np.dot(direction1, direction2)

    if dot1 < 0.6 or dot2 < 0.6 or pair_dot < 0.6:
        return False

    return True

def _detectCheckerboardCorners_(imgFile1, imgFile2, boardSize, debugLevel=1):
    """
    Detect checkerboard corners in a stereo image pair and refine their coordinates.

    Parameters:
        - imgFile1 (str): Path to the left image.
        - imgFile2 (str): Path to the right image.
        - boardSize (tuple): Number of inner corners (rows, cols) on the checkerboard.
        - debugLevel (int): Debug level (0 = silent, 1 = log basic info, 2 = verbose).

    Returns:
        - tuple or None: (corners1, corners2, imgFile1, imgFile2), or None if detection fails.
    """
    # Load stereo image pair from their filenames
    img1 = cv.imread(imgFile1)
    img2 = cv.imread(imgFile2)

    if img1 is None or img2 is None:
        if debugLevel >= 1:
            print(f"Failed to read one or both images:\n{imgFile1}\n{imgFile2}")
        return None

    # Convert the stereo image pair into grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Flags used by checkerboard corner detector
    flags = cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK

    # Detect and store checkerboard corner coordinates
    ret1, corners1 = cv.findChessboardCorners(gray1, boardSize, flags=flags)
    ret2, corners2 = cv.findChessboardCorners(gray2, boardSize, flags=flags)

    if not ret1 or not ret2:
        if debugLevel >= 1:
            print(f"Corners not found for pair: {os.path.basename(imgFile1)} | {os.path.basename(imgFile2)}")
        return None

    # Criteria used by subpixel corner detector
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # Attempt to improve the corner accuracy
    corners_subpix1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    corners_subpix2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

    if debugLevel >= 1:
        print(f"Corners detected: {os.path.basename(imgFile1)} | {os.path.basename(imgFile2)}")

    return (corners_subpix1, corners_subpix2, imgFile1, imgFile2)

def calibrateCheckerboard(boardSize, squareSize, leftImages, rightImages, fileName="chb_caldata.pkl", debugLevel=0):
    """
    Perform stereo camera calibration using a set of checkerboard image pairs.

    Parameters:
        - boardSize (tuple): Number of inner corners per checkerboard row and column (rows, cols).
        - squareSize (float): Real-world size of one square on the checkerboard (e.g. mm or meter).
        - leftImages (list): List of file paths to left camera images.
        - rightImages (list): List of file paths to right camera images.
        - fileName (str): Path to the output pickle file where calibration data will be stored.
        - debugLevel (int): Debug level (0 = silent, 1 = logs, 2 = save debug images and logs).

    Returns:
        - dict: Dictionary containing calibration data (intrinsics, extrinsics, error, etc.).
    """
    # Check if leftImages and rightImages lists are not empty
    if not leftImages or not rightImages:
        raise ValueError("Image lists cannot be empty. Please provide paths to left and right images.")
    # Check if number of leftImages and rightImages are the same
    if len(leftImages) != len(rightImages):
        raise ValueError("The number of left and right images must be the same.")

    if debugLevel >= 1:
        print("\nStarting calibration...")
        start_time = time.time()

    # Prepare the 3D object points for the checkerboard in the real world. For
    # example, if boardSize=(8,6) and squareSize=25mm, it creates a grid from
    # (0,0,0) to (7,5,0)
    rows, cols = boardSize
    objPointGrid = np.zeros((rows * cols, 3), np.float32)
    objPointGrid[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * squareSize

    # Lists to store object points and corresponding image points for valid image pairs
    objPoints = []    # 3D points in real world space
    imgPoints1 = []   # 2D points in left image
    imgPoints2 = []   # 2D points in right image

    # Detect checkerboard corners in all image pairs using multithreading
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        all_results = list(executor.map(
            lambda pair: _detectCheckerboardCorners_(pair[0], pair[1], boardSize, debugLevel),
            zip(leftImages, rightImages)
        ))

    # This will be used to enforce consistent checkerboard orientation across all image pairs
    referenceDirection = [None]

    # If debug images are to be saved, create a directory for output
    if debugLevel == 2:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_dir = f"../debug/corners_{timestamp}_{boardSize[0]}x{boardSize[1]}"
        os.makedirs(debug_dir, exist_ok=True)

    # Process results from corner detection
    for result in all_results:
        if result is None:
            continue

        corners1, corners2, imgFile1, imgFile2 = result

        if debugLevel >= 1:
            print(f"Processing pair: {os.path.basename(imgFile1)} | {os.path.basename(imgFile2)}")

        # Discard pairs with inconsistent checkerboard orientation
        if not _isOrientationConsistent_(corners1, corners2, referenceDirection):
            if debugLevel >= 1:
                print("Inconsistent orientation. Skipping pair.")
            continue

        # Store the valid object/image point sets
        objPoints.append(objPointGrid)
        imgPoints1.append(corners1)
        imgPoints2.append(corners2)

        # Save debug images with drawn corners
        if debugLevel == 2:
            img1 = cv.imread(imgFile1)
            img2 = cv.imread(imgFile2)
            cv.drawChessboardCorners(img1, boardSize, corners1, True)
            cv.drawChessboardCorners(img2, boardSize, corners2, True)
            cv.imwrite(f"{debug_dir}/corners_{os.path.basename(imgFile1)}", img1)
            cv.imwrite(f"{debug_dir}/corners_{os.path.basename(imgFile2)}", img2)

    # If no valid corner pairs were found, raise an error
    if not objPoints:
        raise RuntimeError("No valid corners found in the image set.")

    if debugLevel >= 1:
        print("\nCalibrating individual cameras...")

    # Get image resolution (width, height) from the first image
    img_size = cv.imread(leftImages[0]).shape
    img_size = (img_size[1], img_size[0])  # (width, height)

    # Calibrate each camera individually to get intrinsic parameters
    _, K1, D1, _, _ = cv.calibrateCamera(objPoints, imgPoints1, img_size, None, None)
    _, K2, D2, _, _ = cv.calibrateCamera(objPoints, imgPoints2, img_size, None, None)

    if debugLevel >= 1:
        print("Performing stereo calibration...")

    # Stereo calibration criteria and flags
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    stereo_flags = cv.CALIB_FIX_INTRINSIC + cv.CALIB_SAME_FOCAL_LENGTH

    # Perform stereo calibration to obtain rotation, translation, essential,
    # and fundamental matrices
    err, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        objPoints, imgPoints1, imgPoints2,
        K1, D1, K2, D2,
        img_size,
        criteria=criteria,
        flags=stereo_flags
    )

    # Store all calibration results in a dictionary
    calData = {
        "img_size": img_size,
        "proj_error": err,
        "K1": K1, "D1": D1,           # Left camera intrinsics
        "K2": K2, "D2": D2,           # Right camera intrinsics
        "R": R, "T": T,               # Rotation and translation between cameras
        "E": E, "F": F                # Essential and Fundamental matrices
    }

    # Save calibration data to a file
    with open(fileName, "wb") as f:
        pickle.dump(calData, f)

    if debugLevel >= 1:
        print(f"\nCalibration complete. Results saved to: {fileName}")
        # Print elapsed time for calibration
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        print(f"Elapsed time: {elapsed_seconds:.2f} seconds")

    return calData

def _getParameters_(calData):
    """
    Extract and format stereo calibration data into a flat parameter dictionary.

    This function takes a stereo calibration data dictionary and extracts
    relevant intrinsic, distortion, and extrinsic parameters for both cameras.
    It also converts the rotation matrix to Euler angles (in degrees).

    Parameters:
        - calData (dict): Dictionary containing stereo calibration results. Must include:
            - 'img_size': Tuple of image dimensions (width, height)
            - 'proj_error': Final projection error from stereo calibration
            - 'K1', 'K2': Intrinsic matrices for the left and right cameras
            - 'D1', 'D2': Distortion coefficients (arrays) for left and right cameras
            - 'R': Rotation matrix between the two cameras
            - 'T': Translation vector from left to right camera

    Returns:
        - dict: A flat dictionary with all relevant calibration parameters including:
            - Intrinsics (fx, fy, cx, cy, skew)
            - Distortion coefficients (k1, k2, p1, p2, k3)
            - Extrinsic translation components (Tx, Ty, Tz)
            - Euler angles (Rx, Ry, Rz) representing rotation
    """
    # Basic parameters
    img_size = calData["img_size"]
    proj_error = calData["proj_error"]

    # Intrinsics and distortion for left and right cameras
    K1 = calData["K1"]
    D1 = calData["D1"].squeeze()
    K2 = calData["K2"]
    D2 = calData["D2"].squeeze()

    # Translation and rotation from left to right camera
    T = calData["T"].squeeze()
    R = calData["R"]

    # Convert rotation matrix to Euler angles in degrees (xyz order)
    rotation = Rotation.from_matrix(R)
    Rx, Ry, Rz = rotation.as_euler('xyz', degrees=True)

    # Assemble all parameters into a flat dictionary
    parameters = {
        "img_width" : img_size[0],
        "img_height" : img_size[1],
        "proj_error" : proj_error,

        # Left camera intrinsic parameters
        "K1_fx" : K1[0,0],
        "K1_fy" : K1[1,1],
        "K1_cx" : K1[0,2],
        "K1_cy" : K1[1,2],
        "K1_s"  : K1[0,1],  # Skew is usually 0

        # Left camera distortion coefficients
        "D1_k1" : D1[0],
        "D1_k2" : D1[1],
        "D1_p1" : D1[2],
        "D1_p2" : D1[3],
        "D1_k3" : D1[4],

        # Right camera intrinsic parameters
        "K2_fx" : K2[0,0],
        "K2_fy" : K2[1,1],
        "K2_cx" : K2[0,2],
        "K2_cy" : K2[1,2],
        "K2_s"  : K2[0,1],

        # Right camera distortion coefficients
        "D2_k1" : D2[0],
        "D2_k2" : D2[1],
        "D2_p1" : D2[2],
        "D2_p2" : D2[3],
        "D2_k3" : D2[4],

        # Extrinsic translation from left to right camera
        "Tx" : T[0],
        "Ty" : T[1],
        "Tz" : T[2],

        # Rotation between cameras expressed as Euler angles (xyz order)
        "Rx" : Rx,
        "Ry" : Ry,
        "Rz" : Rz
    }

    return parameters

def printParameters(calData):
    """
    Print calibration parameters extracted from calData in a readable format.

    Parameters:
        - calData (dict): Stereo calibration data.

    Returns:
        - None
    """
    # Extract flattened parameters from calibration dictionary
    parameters = _getParameters_(calData)

    # Create header
    retStr = '\nCalibration Parameters:\n'
    retStr += '------------------------------------------------\n'

    # Loop through parameters and format each key-value pair
    for key in parameters.keys():
        retStr += "  %25s : %s\n" % (key, str(parameters[key]))

    # Output the final formatted parameter list
    print(retStr)


def saveParametersCSV(calData, fileName="calibration_parameters.csv", debugLevel=1):
    """
    Save stereo calibration parameters to a CSV file.

    This function extracts all relevant calibration parameters from the given calibration data
    and writes them to a CSV file in a two-column format: "Parameter", "Value".

    Parameters:
        - calData (dict): Stereo calibration data.
        - fileName (str): Output CSV file path. Default is "calibration_parameters.csv".

    Returns:
        - None
    """
    # Extract flattened parameter dictionary from calibration data
    parameters = _getParameters_(calData)

    # Open the output file for writing
    with open(fileName, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Parameter", "Value"])

        # Write each parameter and its corresponding value
        for key, value in parameters.items():
            writer.writerow([key, value])

    # Print message
    if debugLevel > 0:
        print(f"\nCalibration parameters saved to {fileName}")

def _readParametersCSV_(fileName="calibration_parameters.csv", debugLevel=1):
    """
    Read stereo calibration parameters from a CSV file.

    This function reads a two-column CSV file containing "Parameter" and "Value"
    and reconstructs the parameter dictionary.

    Parameters:
        - fileName (str): Path to the input CSV file.
        - debugLevel (int): Level of verbosity. Default is 1.

    Returns:
        - parameters (dict): Flattened calibration parameter dictionary.
    """
    parameters = {}

    try:
        with open(fileName, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header row

            for row in reader:
                if len(row) != 2:
                    continue  # Skip malformed rows

                key, value = row
                try:
                    # Try to parse numeric values (float or int)
                    num_value = float(value)
                    if num_value.is_integer():
                        num_value = int(num_value)
                    parameters[key] = num_value
                except ValueError:
                    # Fallback to string
                    parameters[key] = value

        if debugLevel > 0:
            print(f"\nCalibration parameters loaded from {fileName}")

        return parameters

    except FileNotFoundError:
        if debugLevel > 0:
            print(f"Error: File '{fileName}' not found.")
        return None

def _getDataFromParameters_(parameters):
    """
    Reconstruct the original calibration data dictionary from flattened parameters.

    Parameters:
        - parameters (dict): Flattened calibration parameter dictionary.

    Returns:
        - calData (dict): Reconstructed `calData` dictionary with keys:
            - 'img_size', 'proj_error', 'K1', 'D1', 'K2', 'D2', 'T', 'R'
    """
    # Intrinsic matrices
    K1 = np.array([
        [parameters["K1_fx"], parameters["K1_s"],  parameters["K1_cx"]],
        [0,                   parameters["K1_fy"], parameters["K1_cy"]],
        [0,                   0,                   1]
    ])

    K2 = np.array([
        [parameters["K2_fx"], parameters["K2_s"],  parameters["K2_cx"]],
        [0,                   parameters["K2_fy"], parameters["K2_cy"]],
        [0,                   0,                   1]
    ])

    # Distortion coefficients (assumed 5 elements)
    D1 = np.array([[
        parameters["D1_k1"],
        parameters["D1_k2"],
        parameters["D1_p1"],
        parameters["D1_p2"],
        parameters["D1_k3"]]
    ])

    D2 = np.array([[
        parameters["D2_k1"],
        parameters["D2_k2"],
        parameters["D2_p1"],
        parameters["D2_p2"],
        parameters["D2_k3"]]
    ])

    # Translation vector
    T = np.array([[parameters["Tx"]],
                  [parameters["Ty"]],
                  [parameters["Tz"]]])

    # Rotation matrix from Euler angles (xyz order, degrees)
    rotation = Rotation.from_euler('xyz', [parameters["Rx"], parameters["Ry"], parameters["Rz"]], degrees=True)
    R = rotation.as_matrix()

    # Image size
    img_size = (parameters["img_width"], parameters["img_height"])

    # Rebuild the calibration dictionary
    calData = {
        "img_size": img_size,
        "proj_error": parameters["proj_error"],
        "K1": K1,
        "D1": D1,
        "K2": K2,
        "D2": D2,
        "T": T,
        "R": R
    }

    return calData

def getDataFromParametersCSV(fileName="calibration_parameters.csv"):
    """
    Load stereo calibration data from a CSV file.

    This function reads a CSV file containing flattened stereo calibration parameters,
    reconstructs the original calibration data structure (`calData`) by internally calling:
        - _readParametersCSV_: to read the CSV into a flat parameter dictionary
        - _getDataFromParameters_: to reconstruct the full calibration dictionary

    Parameters:
        - fileName (str): Path to the CSV file containing the calibration parameters.
                          Default is 'calibration_parameters.csv'.

    Returns:
        - calData (dict): Dictionary containing stereo calibration data.
    """
    calData = _getDataFromParameters_(_readParametersCSV_(fileName))
    return calData

def loadData(fileName):
    """
    Load stereo calibration data from a pickle (.pkl) file.

    This function attempts to load a Python dictionary containing stereo calibration
    data (as produced by `calibrateCheckerboard`) from a pickle file.

    Parameters:
        - fileName (str): Path to the pickle (.pkl) file containing the calibration data.

    Returns:
        - calData (dict): Dictionary containing stereo calibration data if successful.

    Raises:
        - FileNotFoundError: If the file does not exist.
        - pickle.UnpicklingError: If the file cannot be unpickled properly.
        - Exception: For any other unexpected errors during file loading.
    """
    try:
        with open(fileName, "rb") as file:
            calData = pickle.load(file)
            print(f"Successfully loaded calibration data from '{fileName}'")
            return calData

    except FileNotFoundError:
        print(f"Error: File '{fileName}' not found.")
        raise

    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle the file '{fileName}'. Ensure it is a valid pickle file.")
        raise

    except Exception as e:
        print(f"Error: An unexpected error occurred while loading '{fileName}': {e}")
        raise
