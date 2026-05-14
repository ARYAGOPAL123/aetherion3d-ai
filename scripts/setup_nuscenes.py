"""Download and setup nuScenes mini dataset for evaluation."""
import subprocess
import sys
from pathlib import Path


def main():
    print("Installing nuScenes devkit...")
    subprocess.run([sys.executable, "-m", "pip", "install", "nuscenes-devkit", "pillow"], check=True)
    
    print("\nDownloading nuScenes mini version (300MB)...")
    print("Visit https://www.nuscenes.org/nuscenes to accept the license.")
    print("\nThen run:")
    print("  python -c \"from nuscenes.nuscenes import NuScenes; NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)\"")
    
    data_dir = Path("data/nuscenes")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nData directory ready: {data_dir.absolute()}")


if __name__ == "__main__":
    main()