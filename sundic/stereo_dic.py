import copy
import glob
import os
import shutil
from enum import IntEnum

import cv2 as cv
import matplotlib.pyplot as plt
import natsort as ns
import numpy as np
import sundic.sundic as sdic
from sundic.sundic import CompID, IntConst
from scipy.interpolate import griddata
import sundic.post_process as sdpp
from skimage.exposure import match_histograms
import ray as ray
from concurrent.futures import ThreadPoolExecutor

from sundic.util.fast_interp import interp2d
import sundic.util.datafile as dataFile
from scipy.interpolate import NearestNDInterpolator
from sundic.util.savitsky_golay import sgolay2d
from skimage.exposure import match_histograms

# Modify the subset array indices to include z-coordinates and diplacements
class CompID(IntEnum):
    XCoordID = 0   # The x-coordinate of the subset center point
    YCoordID = 1   # The y-coordinate of the subset center point
    SSSizeID = 2   # The subset size
    ShapeFnID = 3   # The shape function - 0 = affine, 1 = quadratic
    CZNSSDID = 4   # The CZNSSD value for the subset
    XDispID = 5   # The x-displacement of the subset point - start of x model coefficients
    YDispID = 11  # The y-displacement of the subset point - start of y model coefficients
    ZCoordID = 17   # The z-coordinate of the subset center point
    ZDispID = 18   # The z-displacement of the subset point - start of z model coefficients

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
            if os.path.splitext(file)[1].lower() in supported_file_types
            and os.path.splitext(file)[0].endswith('_0')
        ])
        # Right image files end with _1
        right_files = ns.natsorted([
            file for file in os.listdir(root_path)
            if os.path.splitext(file)[1].lower() in supported_file_types
            and os.path.splitext(file)[0].endswith('_1')
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

from scipy.spatial.transform import Rotation as Rscipy

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
    pts1 = cv.undistortPoints(leftPts, K1, D1)
    pts2 = cv.undistortPoints(rightPts, K2, D2)

    # Reshape to (2, N) for triangulation
    pts1 = pts1.reshape(-1, 2).T
    pts2 = pts2.reshape(-1, 2).T

    # Build projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))     # [I | 0]
    P2 = np.hstack((R, T))                            # [R | T]

    # Triangulate homogeneous 3D points
    points_hom = cv.triangulatePoints(P1, P2, pts1, pts2)  # shape: (4, N)

    # Convert from homogeneous coordinates to 3D
    points_3d = cv.convertPointsFromHomogeneous(points_hom.T)

    return points_3d.squeeze()

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
    # 1. Extract 2D Points
    # Extract the (x, y) coordinates from the subset point arrays
    left_pts_2d = np.stack((
        leftSubSetPnts[:, :, CompID.XCoordID].flatten(),
        leftSubSetPnts[:, :, CompID.YCoordID].flatten()
    ), axis=1)

    right_pts_2d = np.stack((
        rightSubSetPnts[:, :, CompID.XCoordID].flatten(),
        rightSubSetPnts[:, :, CompID.YCoordID].flatten()
    ), axis=1)

    # 2. Triangulate to get 3D points
    points_3d = _triangulatePoints_(left_pts_2d, right_pts_2d, calData)

    # Check if triangulation returned valid points
    # if points_3d is None or points_3d.ndim != 2 or points_3d.shape[1] != 3:
    #     print("Warning: Triangulation did not return valid 3D points. Z-coordinates not updated.")
    #     return leftSubSetPnts, rightSubSetPnts

    # Create worldSubSetPnts with correct shape to store z-coordinates and displacements
    subset_grid_shape = (leftSubSetPnts.shape[0], leftSubSetPnts.shape[1], int(CompID.ZDispID) + 1)
    worldSubSetPnts = np.zeros(shape=subset_grid_shape)

    # Update the CoordID fields
    data_shape = leftSubSetPnts[:, :, CompID.XCoordID].shape
    worldSubSetPnts[:, :, CompID.XCoordID] = points_3d[:, 0].reshape(data_shape)
    worldSubSetPnts[:, :, CompID.YCoordID] = points_3d[:, 1].reshape(data_shape)
    worldSubSetPnts[:, :, CompID.ZCoordID] = points_3d[:, 2].reshape(data_shape)

    # Reset correlation metric
    worldSubSetPnts[:, :, CompID.CZNSSDID] = IntConst.CNZSSD_MAX

    # Store the subset size
    worldSubSetPnts[:, :, CompID.SSSizeID] = leftSubSetPnts[:, :, CompID.SSSizeID]

    # Store the shape function type
    worldSubSetPnts[:, :, CompID.ShapeFnID] = leftSubSetPnts[:, :, CompID.ShapeFnID]

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
    for coord_id in [sdic.CompID.XCoordID, sdic.CompID.YCoordID]:
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

def _resizeSubSetArray_(subset_array, required_size):
    """
    Resizes a subset point array to a required size if it's too small.

    This helper function checks the size of the last dimension of the input array.
    If it's smaller than the required size, it creates a new, larger array,
    copies the old data into it, and returns the new array. Otherwise, it
    returns the original array unchanged.

    Parameters:
        subset_array (ndarray): The input subset point array to check and resize.
        required_size (int): The minimum required size for the last dimension.

    Returns:
        ndarray: The resized (or original) subset point array.
    """
    current_shape = subset_array.shape
    # Check if the last dimension is smaller than what's required
    if current_shape[2] < required_size:
        # print(f"Resizing subset array from shape {current_shape} to a new size of "
        #       f"({current_shape[0]}, {current_shape[1]}, {required_size}).")

        # Define the shape for the new, larger array
        new_shape = (current_shape[0], current_shape[1], required_size)

        # Create a new array filled with zeros
        resized_array = np.zeros(new_shape, dtype=subset_array.dtype)

        # Copy the data from the smaller, original array into the new one
        resized_array[:, :, :current_shape[2]] = subset_array

        return resized_array

    # If no resize is needed, return the original array
    return subset_array

def stereoMatch(settings, leftImg, rightImg, fillMissing=False):
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
    # histMatch=True
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
    sm_settings.CPUCount = 1       # Multiproccessing currently broken for stereo matching
    sm_settings.ReferenceStrategy = "Absolute"
    sm_settings.DatumImage = 0
    sm_settings.TargetImage = -1
    sm_settings.Increment = 1
    resultsPath = os.path.join(tempFolderPath, "sm_results.sdic")

    # Perform planar DIC to get subset coordinates and displacements
    returnData = sdic.planarDICLocal(sm_settings, resultsPath)
    rightSubSetPnts = np.copy(returnData[0])
    leftSubSetPnts = np.copy(returnData[0])

    # Optional debug output for subset matching statistics
    if sm_settings.DebugLevel >= 2:
        results, nRows, nCols = sdpp.getDisplacements(resultsPath, -1)
        results = results[~np.isnan(results).any(axis=1)]
        foundPoints = results.shape[0]
        totalPoints = nRows * nCols
        print(f"Found {foundPoints}/{totalPoints} subsets. Missing: {totalPoints - foundPoints}")

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
    if fillMissing:
        rightSubSetPnts = _fillMissingSubsets_(rightSubSetPnts)
        if sm_settings.DebugLevel >= 1:
            print("Filled in missing subsets for right image.")

    # Force all the subset point coordinates to be ints
    # Warning, may cause errors. Update: Did cause major errors
    # rightSubSetPnts[:, :, CompID.XCoordID] = np.round(rightSubSetPnts[:, :, CompID.XCoordID])
    # rightSubSetPnts[:, :, CompID.YCoordID] = np.round(rightSubSetPnts[:, :, CompID.YCoordID])

    # 4. Cleanup
    if sm_settings.DebugLevel < 2:
        shutil.rmtree(tempFolderPath)

    return leftSubSetPnts, rightSubSetPnts

def temporalMatch(initSubSetPnts, imgSet, settings, resultsFile, externalRay=False, guiThread=None):
    """
    Perform local planar (2D) Digital Image Correlation (DIC) analysis.

    This function takes a dictionary of settings as input and performs local DIC analysis
    based on the specified settings. The analysis involves processing a series of image pairs
    to obtain displacement and strain data.

    Parameters:
        - settings: A Settings object containing the settings for the DIC analysis.
        - resultsFile: The name of the file to store the results in.
        - externalRay: A boolean indicating whether to use an external ray server or not.
        - guiThread: The GUI thread object if running from the GUI, otherwise None. Used to
                    cleanly stop the analysis if requested from the GUI.

    Returns:
        - returnData (list): A list of subSetPoint arrays. Each subSetPoint array is a
            3D matrix where the first plane contains the x-coordinates
            the second plane the y-coordinates and the remaining planes the subset size,
            shapeFn, CZNSSD value and model coefficients.  This array can be processed to
            obtain displacement and strain data and to generate graphs.

    Raises:
        - ValueError: If an invalid optimization algorithm is specified.
    """
    try:
        # Let's set a random seed for repeatable results
        np.random.seed(42)

        # Store the debug level
        debugLevel = settings.DebugLevel

        # Define measurement points using the settings specified in the config file
        # These are the center points of the subsets
        subSetSize = settings.SubsetSize
        stepSize = settings.StepSize
        shapeFn = settings.ShapeFunctions
        subSetPnts = initSubSetPnts # Change from planarDICLocal

        # Deal with a binary mask if specified
        roiMask = None
        activeSubsets = np.ones(subSetPnts.shape[:2], dtype=bool)

        if settings.hasMask():
            img0 = sdic.readImage(imgSet[0])
            roiMask = sdic._loadMask_(settings.MaskFile, img0.shape)
            activeSubsets = sdic._buildActiveSubsetsMask_(subSetPnts, roiMask)

            if debugLevel > 0:
                nActive = np.count_nonzero(activeSubsets)
                nTotal = activeSubsets.size
                print('\nMask Information :')
                print('---------------------------------')
                print(f'  Active subsets   : {nActive}')
                print(f'  Inactive subsets : {nTotal - nActive}')

        if not np.any(activeSubsets):
            raise ValueError("The specified mask excludes all subset centers. No active subsets remain.")

        # Get the image pair information
        imgDatum = settings.DatumImage
        imgTarget = settings.TargetImage
        if imgTarget == -1:
            imgTarget = len(imgSet)-1
        imgIncr = settings.Increment
        imgPairs = int((imgTarget - imgDatum)/imgIncr)

        # Debug output if requested
        if debugLevel > 0:
            print('\nImage Pair Information :')
            print('---------------------------------')
            print('  Number of image pairs : {}'.format(imgPairs))

        # Setup serialization of the data to msgpack binary file
        df = dataFile.DataFile.openWriter(resultsFile)
        df.writeHeading(settings)

        # Initialize the parallel enviroment if required
        nCpus = settings.CPUCount
        if nCpus > 1:
            if debugLevel > 0:
                print('\nParallel Run Information :')
                print('---------------------------------')
                print('  Starting parallel run with {} CPUs'.format(nCpus))
                if externalRay:
                    print('  Using external ray server')

                # Init ray with restarts
                sdic._safeRayInit_(externalRay, nCpus, debugLevel=debugLevel)

        # Loop through all image pairs to perform the local DIC
        returnData = []
        x_coordInit = np.copy(subSetPnts[:, :, CompID.XCoordID])
        y_coordInit = np.copy(subSetPnts[:, :, CompID.YCoordID])

        for imgPairIdx, img in enumerate(range(imgDatum, imgTarget, imgIncr)):

            # Store previous iteration displacement values
            x_dispPrev = np.copy(subSetPnts[:, :, CompID.XDispID])
            y_dispPrev = np.copy(subSetPnts[:, :, CompID.YDispID])

            # Setup the parallel run and wait for all results
            if nCpus > 1:

                ray = sdic._require_ray()
                _rmt_icOptimization_ = sdic._get_rmt_icOptimization()

                # Turn of debugging temporarily
                nDebugOld = settings.DebugLevel
                settings.DebugLevel = 0

                # Setup the submatrices - match shape to image if possible
                nTotRows, nTotCols, _ = subSetPnts.shape
                mRows, mCols = _factorCPUCount_(nCpus, nTotRows/nTotCols)
                if nDebugOld > 0:
                    print("\n  Splitting matrix into {}x{} submatrices".format(
                        mRows, mCols))
                    print("")
                subMatrices = _splitMatrix_(subSetPnts, mRows, mCols)
                activeSubMatrices = _splitMatrix_(activeSubsets, mRows, mCols)

                # Track the processes that are being submitted
                procIDs = []
                for i in range(mRows*mCols):
                    iRow, iCol = np.unravel_index(i, (mRows, mCols))
                    procIDs.append(_rmt_icOptimization_.remote(
                        settings, iRow, iCol, subMatrices[iRow][iCol],
                        activeSubMatrices[iRow][iCol], imgSet, img, guiThread=guiThread))

                    if nDebugOld > 0:
                        print("  Starting remote process for submatrix {} {}".
                              format(iRow, iCol))

                if nDebugOld > 0:
                    print("")

                # Wait for results - start pulling results from tasks as soon as they are
                # are done
                while len(procIDs):
                    done_id, procIDs = ray.wait(procIDs)

                    # Launch ray tasks with retries
                    iRow, iCol, rsltMatrix = sdic._safeRayLaunch_(
                        done_id[0], debugLevel=nDebugOld)
                    (subMatrices[iRow][iCol])[:] = rsltMatrix
                    if nDebugOld > 0:
                        print("  Submatrix {} {} completed".format(iRow, iCol))

                # Turn debugging back on
                settings.DebugLevel = nDebugOld

            # Serial run on one processor
            else:
                # coefficients at convergence for current (i'th) image pair
                subSetPnts[:] = sdic._icOptimization_(
                    settings, subSetPnts, activeSubsets, imgSet, img, guiThread=guiThread)

            # Update the subset points coordinates if required - we make copies of the
            # current subset points to create a new array of subset points
            if settings.isRelativeStrategy():
                subSetPnts[:] = sdic._updateSubSets_(x_coordInit, y_coordInit, x_dispPrev, y_dispPrev,
                                                subSetPnts)

            # Store the current subset points in the return data
            subSetPntsOut = np.copy(subSetPnts)
            subSetPntsOut[:, :, CompID.XCoordID] = x_coordInit
            subSetPntsOut[:, :, CompID.YCoordID] = y_coordInit
            subSetPntsOut = _applyInactiveSubsets_(subSetPntsOut, activeSubsets)
            returnData.append(subSetPntsOut)
            df.writeSubSetData(imgPairIdx, subSetPntsOut)

            # Make some debug output
            if (settings.DebugLevel > 0):
                print('\n  ------------------------------------------------------')
                print('  Image pair {} processed:'.format(imgPairIdx))
                if settings.isAbsoluteStrategy():
                    print('    '+imgSet[imgDatum])
                else:
                    print('    '+imgSet[img])
                print('    '+imgSet[img+imgIncr])
                print('  ------------------------------------------------------\n')

        # Shutdown the parallel environment if required
        if settings.CPUCount > 1:
            sdic._safeRayShutdown_(externalRay, debugLevel=debugLevel)

        # Close the file
        df.close()

        return returnData

    # Handle exceptions and shutdown ray if required
    except Exception as e:
        if settings.CPUCount > 1:
            sdic._safeRayShutdown_(externalRay, debugLevel=debugLevel)
        raise e

# TODO: Add option to set calibration parameters location in settings file. The
# parameters should be stored in a csv file in the required format.
# For now, first get the calibration data outside the function by performing
# stereo calibration or read it from a csv file using sc.getDataFromParametersCSV(calCSV)
# TODO: Add option in settings to fillMissing subset points
# TODO: Use ray to run temporalMatch in parallel
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
    leftSubSetPnts_init, rightSubSetPnts_init = stereoMatch(settings, refLeftImg, refRightImg, fillMissing)

    # 4. Perform temporal matching on the left image set
    if settings.DebugLevel > 0:
        print("Performing temporal matching on the left image set...")
    results_left = temporalMatch(leftSubSetPnts_init, leftImgSet, settings, "tm_results_left.sdic")

    # 5. Perform temporal matching on the right image set
    if settings.DebugLevel > 0:
        print("Performing temporal matching on the right image set...")
    results_right = temporalMatch(rightSubSetPnts_init, rightImgSet, settings, "tm_results_right.sdic")

    # 6. Perform coordinate system transform to transform the displacements from
    #    the left image plane and right image plane (2D) to the world
    #    coordinates (3D)

    # Get the temporal image pair information
    imgDatum = settings.DatumImage
    imgTarget = settings.TargetImage
    if imgTarget == -1:
        imgTarget = len(leftImgSet)-1
    imgIncr = settings.Increment
    imgPairs = int((imgTarget - imgDatum)/imgIncr)

    # Prepare results file writer for stereo results
    df = dataFile.DataFile.openWriter(resultsFile)
    df.writeHeading(settings)

    returnData = []
    # Initial triangulation — always from imgDatum
    worldSubSetPnts_ref = _triangulateSubSets_(leftSubSetPnts_init, rightSubSetPnts_init, calData)
    worldSubSetPnts_out = np.copy(worldSubSetPnts_ref)

    for imgPairIdx, img in enumerate(range(imgDatum, imgTarget, imgIncr)):
        print(f"imgPairIdx: {imgPairIdx}, img: {img}")
        # For each stereo image pair, perform least-squares triangulation on the
        # left and right subset points to get the world subset points
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
        # worldSubSetPnts = _triangulateSubSets_(leftSubSetPnts, rightSubSetPnts, calData)

        # Triangulate current stereo pair
        worldSubSetPnts = _triangulateSubSets_(leftSubSetPnts, rightSubSetPnts, calData)

        # Calculate displacements from the reference (either absolute or previous)
        worldSubSetPnts_out[:, :, CompID.XDispID] = worldSubSetPnts[:, :, CompID.XCoordID] - worldSubSetPnts_ref[:, :, CompID.XCoordID]
        worldSubSetPnts_out[:, :, CompID.YDispID] = worldSubSetPnts[:, :, CompID.YCoordID] - worldSubSetPnts_ref[:, :, CompID.YCoordID]
        worldSubSetPnts_out[:, :, CompID.ZDispID] = worldSubSetPnts[:, :, CompID.ZCoordID] - worldSubSetPnts_ref[:, :, CompID.ZCoordID]

        # Update coordinates from triangulation
        worldSubSetPnts_out[:, :, CompID.XCoordID] = worldSubSetPnts[:, :, CompID.XCoordID]
        worldSubSetPnts_out[:, :, CompID.YCoordID] = worldSubSetPnts[:, :, CompID.YCoordID]
        worldSubSetPnts_out[:, :, CompID.ZCoordID] = worldSubSetPnts[:, :, CompID.ZCoordID]

        # TODO: Ask Prof. Venter why the subset coordinates are always set to
        # their initial positions?

        if settings.isRelativeStrategy():
            # Update the reference subSetPnts when using relative strategy
            worldSubSetPnts_ref = np.copy(worldSubSetPnts)
            # Update coordinates from triangulation
            # worldSubSetPnts[:, :, CompID.XCoordID] = worldSubSetPnts_new[:, :, CompID.XCoordID]
            # worldSubSetPnts[:, :, CompID.YCoordID] = worldSubSetPnts_new[:, :, CompID.YCoordID]
            # worldSubSetPnts[:, :, CompID.ZCoordID] = worldSubSetPnts_new[:, :, CompID.ZCoordID]

        # Write this frame's results to the results file
        df.writeSubSetData(imgPairIdx, worldSubSetPnts_out)

        returnData.append(worldSubSetPnts_out)

    df.close()

    return returnData

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
        plt.show(fig)
    else:
        plt.close(fig)

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
        worldSubSetPnts[:, :, CompID.XCoordID].flatten(),
        worldSubSetPnts[:, :, CompID.YCoordID].flatten(),
        worldSubSetPnts[:, :, CompID.ZCoordID].flatten()
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
        plt.show(fig)
    else:
        plt.close(fig)

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
        "x": CompID.XDispID,
        "y": CompID.YDispID,
        "z": CompID.ZDispID,
    }

    # Extract 3D coordinates
    x = worldSubSetPnts[:, :, CompID.XCoordID].flatten()
    y = worldSubSetPnts[:, :, CompID.YCoordID].flatten()
    z = worldSubSetPnts[:, :, CompID.ZCoordID].flatten()

    # Get displacements
    dx = worldSubSetPnts[:, :, CompID.XDispID].flatten()
    dy = worldSubSetPnts[:, :, CompID.YDispID].flatten()
    dz = worldSubSetPnts[:, :, CompID.ZDispID].flatten()

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
    sc = ax.scatter(x, y, z, c=disp, cmap='viridis', s=8, alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
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
