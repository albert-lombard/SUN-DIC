# Regression Test Framework

# Currently tests the outputs of planarDICLocal, getDisplacements and
# getStrains.

# Example use cases:
#
# Compare the current commit with the previous:
#       python regression.py compare-commits HEAD HEAD^1
#
# Force regeneration of baselines:
#       python regression.py compare-commits HEAD HEAD^1 --force
#
# Compare the current uncommitted code to the current commit:
#       python regression.py compare-current
#
# or to a specific commit:
#       python regression.py compare-current <commit-id>

import os
import sys
import shutil
import pathlib
import argparse
import subprocess

# Set paths relative to this script's location
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent.resolve()
CACHE_DIR = SCRIPT_DIR / "cache"

# Tolerance values: |val - exp| ≤ ATOL + RTOL × |exp|
RTOL = 1e-3
ATOL = 1e-5


def run_cmd(cmd, cwd=REPO_ROOT, ignore_errors=False):
    """
    Helper function for running shell commands.
    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if not ignore_errors and result.returncode != 0:
        print(f"Command failed:\n{result.stderr}")
        sys.exit(result.returncode)
    return result.stdout.strip()


def resolve_commit(commit_ref):
    """
    Convert a git reference (like HEAD^1) to a full commit hash.
    """
    return run_cmd(['git', 'rev-parse', commit_ref])


def generate_local_baselines(out_dir):
    """
    Run DIC analysis for the current workspace and save results to out_dir.
    """
    # Import locally so we don't crash the CLI if dependencies are missing on
    # the host
    import pandas as pd
    import sundic.sundic as sdic
    import sundic.post_process as sdpp
    import sundic.settings as sdset

    # Create output directory
    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Change cwd to this script's directory
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        # Set file paths
        settings_filename = 'settings.ini'
        results_filepath = str(out_dir / 'baseline_results.sdic')

        # Get settings object from settings file
        settings = sdset.Settings.fromSettingsFile(settings_filename)

        # Perform local planar DIC analysis to generate results file
        sdic.planarDICLocal(settings, results_filepath)

        # Calculate and export displacements from results file
        disp, _, _ = sdpp.getDisplacements(results_filepath, -1, smoothWindow=15)
        pd.DataFrame({
            'X_COORD' : disp[:, sdpp.CompID.XCoordID],
            'Y_COORD' : disp[:, sdpp.CompID.YCoordID],
            'X_DISP'  : disp[:, sdpp.DispComp.X_DISP],
            'Y_DISP'  : disp[:, sdpp.DispComp.Y_DISP],
            'DISP_MAG': disp[:, sdpp.DispComp.DISP_MAG]
        }).to_pickle(out_dir / 'baseline_displacements.pkl')

        # Calculate export strains from results file
        strains, _, _ = sdpp.getStrains(results_filepath, -1, smoothWindow=15)
        pd.DataFrame({
            'X_COORD'     : strains[:, sdpp.CompID.XCoordID],
            'Y_COORD'     : strains[:, sdpp.CompID.YCoordID],
            'X_STRAIN'    : strains[:, sdpp.StrainComp.X_STRAIN],
            'Y_STRAIN'    : strains[:, sdpp.StrainComp.Y_STRAIN],
            'SHEAR_STRAIN': strains[:, sdpp.StrainComp.SHEAR_STRAIN],
            'VM_STRAIN'   : strains[:, sdpp.StrainComp.VM_STRAIN]
        }).to_pickle(out_dir / 'baseline_strains.pkl')

        print(f"Baselines successfully written to {out_dir}")
    finally:
        # Change back to original directory
        os.chdir(original_cwd)


def generate_baselines(commit_id, force=False):
    """
    Generate baselines for a given commit using an isolated Git worktree.
    """
    # Get commit hash and set its cache directory
    full_commit = resolve_commit(commit_id)
    commit_cache_dir = CACHE_DIR / full_commit

    # Check if cache exists and has the required files
    if not force and commit_cache_dir.exists() and len(list(commit_cache_dir.glob("baseline_*"))) >= 3:
        print(f"Baselines for {full_commit[:8]} already cached. Skipping generation.")
        return commit_cache_dir

    # Refresh cache if --force flag is used
    if force and commit_cache_dir.exists():
        print(f"Clearing existing cache for {full_commit[:8]} due to --force flag.")
        shutil.rmtree(commit_cache_dir)

    # Display short commit ID
    print(f"Generating baselines for commit {full_commit[:8]}...")

    # Set path for new temporary worktree
    worktree_dir = REPO_ROOT / f".worktree_{full_commit[:8]}"

    # Create commit cache directory
    commit_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create a new worktree for the commit
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir)
        run_cmd(['git', 'worktree', 'add', str(worktree_dir), full_commit])

        # Setup virtual environment and install dependencies in the worktree
        venv_dir = worktree_dir / ".test_venv"
        run_cmd([sys.executable, '-m', 'venv', str(venv_dir)])

        # Find the correct python executable
        bin_dir = "Scripts" if os.name == 'nt' else "bin"
        python_exe = venv_dir / bin_dir / "python"

        # Install SUN-DIC and its dependencies in venv
        print("Installing dependencies...")
        run_cmd([str(python_exe), '-m', 'pip', 'install', '-e', '.'],
                cwd=worktree_dir)

        # Copy regression.py, settings.ini and planar_images to worktree.
        # This ensure latest version of regression test is used.

        # script directory
        worktree_script_dir = worktree_dir / "tests" / "planar"
        worktree_script_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(SCRIPT_DIR / "regression.py", worktree_script_dir / "regression.py")
        shutil.copy2(SCRIPT_DIR / "settings.ini", worktree_script_dir / "settings.ini")

        planar_images_dest = worktree_script_dir / "planar_images"
        if planar_images_dest.exists():
            shutil.rmtree(planar_images_dest)
        shutil.copytree(SCRIPT_DIR / "planar_images", planar_images_dest)

        # Call this script from inside the temporary worktree to generate the baselines
        print("Generating baseline results from within worktree...")
        script_in_worktree = worktree_dir / "tests" / "planar" / "regression.py"
        run_cmd([str(python_exe), str(script_in_worktree), 'generate-local',
                 str(commit_cache_dir)], cwd=worktree_dir)

    finally:
        # Remove the worktree
        if worktree_dir.exists():
            run_cmd(['git', 'worktree', 'remove', '--force', str(worktree_dir)])

    return commit_cache_dir


def compare_results(dir_a, dir_b):
    """
    Compare baseline files between two directories.
    """
    import pandas as pd
    import numpy as np
    import sundic.util.datafile as dataFile

    print(f"Comparing {dir_a.name} against {dir_b.name}...")

    # Set flag
    passed = True

    # Compare raw subset data (results.sdic)
    print("Checking subset data...   ", end="")

    # Read data files and get subset data for last image (-1)
    file_a = dataFile.DataFile.openReader(dir_a / 'baseline_results.sdic')
    res_a = dataFile.DataFile.readSubSetData(file_a, -1)
    file_b = dataFile.DataFile.openReader(dir_b / 'baseline_results.sdic')
    res_b = dataFile.DataFile.readSubSetData(file_b, -1)

    # Catch error
    try:
        np.testing.assert_allclose(res_a, res_b, rtol=RTOL, atol=ATOL)
        print("[PASS]")
    except AssertionError as e:
        print("[FAIL]")
        print(f'{str(e)}\n')
        passed = False

    # Compare displacements
    print("Checking displacements... ", end="")

    # Read dataframes for displacements
    df_disp_a = pd.read_pickle(dir_a / 'baseline_displacements.pkl').to_numpy()
    df_disp_b = pd.read_pickle(dir_b / 'baseline_displacements.pkl').to_numpy()

    # # Find rows that have NaNs in either array
    # nan_mask = np.isnan(df_disp_a).any(axis=1) | np.isnan(df_disp_b).any(axis=1)
    # # Remove rows with NaN values
    # df_disp_a = df_disp_a[~nan_mask]
    # df_disp_b = df_disp_b[~nan_mask]

    # # Count total NaNs in both arrays
    # nan_count_a = np.isnan(df_disp_a).sum()
    # nan_count_b = np.isnan(df_disp_b).sum()
    # print(f"Total NaNs found - df_disp_a: {nan_count_a}, df_disp_b: {nan_count_b}")
    # # Count rows to be removed
    # print(f"Rows removed: {nan_mask.sum()}")

    # Catch error
    try:
        np.testing.assert_allclose(df_disp_a, df_disp_b, rtol=RTOL, atol=ATOL)
        print("[PASS]")
    except AssertionError as e:
        print("[FAIL]")
        print(f'{str(e)}\n')
        passed = False

    # Compare Strains
    print("Checking strains...       ", end="")

    # Read dataframes for strains
    df_str_a = pd.read_pickle(dir_a / 'baseline_strains.pkl').to_numpy()
    df_str_b = pd.read_pickle(dir_b / 'baseline_strains.pkl').to_numpy()

    # # Find rows that have NaNs in either array
    # nan_mask = np.isnan(df_str_a).any(axis=1) | np.isnan(df_str_b).any(axis=1)
    # # Remove rows with NaN values
    # df_str_a = df_str_a[~nan_mask]
    # df_str_b = df_str_b[~nan_mask]

    # # Count total NaNs in both arrays
    # nan_count_a = np.isnan(df_str_a).sum()
    # nan_count_b = np.isnan(df_str_b).sum()
    # print(f"Total NaNs found - df_str_a: {nan_count_a}, df_str_b: {nan_count_b}")
    # # Count rows to be removed
    # print(f"Rows removed: {nan_mask.sum()}")

    # Catch error
    try:
        np.testing.assert_allclose(df_str_a, df_str_b, rtol=RTOL, atol=ATOL)
        print("[PASS]")
    except AssertionError as e:
        print("[FAIL]")
        print(f'{str(e)}\n')
        passed = False

    # Pass test if all comparisons are equal
    if passed:
        print("\nAll results match within tolerance.")
    else:
        print("\nRegression detected: Some results did not match.")

    return passed


def compare_commits(commit_a, commit_b, force=False):
    """
    Generate baselines for two commits and compare them.
    """
    # Generate baseline results if they don't exist, else use cached results
    dir_a = generate_baselines(commit_a, force=force)
    dir_b = generate_baselines(commit_b, force=force)

    # Compare commit cache directories
    compare_results(dir_a, dir_b)


def compare_current_to_commit(commit_id, force=False):
    """
    Compare the current working directory to a specific commit.
    """
    # Generate baseline results for commit
    commit_dir = generate_baselines(commit_id, force=force)

    # Generate baseline results for current code
    print("\nGenerating baselines for CURRENT working directory...")
    current_cache_dir = CACHE_DIR / "current_workspace"
    generate_local_baselines(current_cache_dir)

    # Compare current code and commit directories
    compare_results(current_cache_dir, commit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Internal Command: generate-local (called by the worktree)
    p_loc = subparsers.add_parser("generate-local", help=argparse.SUPPRESS)
    p_loc.add_argument("out_dir")

    # Command: generate
    p_gen = subparsers.add_parser("generate", help="Generate and cache baselines for a commit")
    p_gen.add_argument("commit", help="Commit hash or reference (e.g., HEAD^1)")
    p_gen.add_argument("--force", action="store_true", help="Force regeneration of baselines")

    # Command: compare-commits
    p_cc = subparsers.add_parser("compare-commits", help="Compare two historical commits")
    p_cc.add_argument("commit_a", help="First commit reference")
    p_cc.add_argument("commit_b", help="Second commit reference")
    p_cc.add_argument("--force", action="store_true", help="Force regeneration of baselines")

    # Command: compare-current
    p_cur = subparsers.add_parser("compare-current",
                                  help="Compare unstaged workspace against a commit")
    p_cur.add_argument("commit", nargs="?", default="HEAD",
                       help="Target commit to compare against (default: HEAD)")
    p_cur.add_argument("--force", action="store_true", help="Force regeneration of baselines")

    args = parser.parse_args()

    if args.command == "generate-local":
        generate_local_baselines(args.out_dir)
    elif args.command == "generate":
        generate_baselines(args.commit, force=args.force)
    elif args.command == "compare-commits":
        compare_commits(args.commit_a, args.commit_b, force=args.force)
    elif args.command == "compare-current":
        compare_current_to_commit(args.commit, force=args.force)
