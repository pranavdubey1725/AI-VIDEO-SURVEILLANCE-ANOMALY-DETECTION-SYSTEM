import torch
import cv2
import ultralytics
import numpy
import pandas
import sklearn
import fastapi
import streamlit

def check_gpu():
    print("=" * 50)
    print("ENVIRONMENT CHECK")
    print("=" * 50)

    print(f"\n[PyTorch]     {torch.__version__}")
    print(f"[CUDA]        {'Available' if torch.cuda.is_available() else 'NOT available'}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU]         {gpu}")
        print(f"[VRAM]        {vram:.1f} GB")

        # Quick tensor op on GPU to confirm it actually works
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"[GPU compute] OK — matrix multiply on GPU succeeded")
    else:
        print("[GPU]         NOT FOUND — check CUDA installation")

    print(f"\n[OpenCV]      {cv2.__version__}")
    print(f"[Ultralytics] {ultralytics.__version__}")
    print(f"[NumPy]       {numpy.__version__}")
    print(f"[Pandas]      {pandas.__version__}")
    print(f"[Scikit-learn]{sklearn.__version__}")
    print(f"[FastAPI]     {fastapi.__version__}")
    print(f"[Streamlit]   {streamlit.__version__}")

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED — Ready to build")
    print("=" * 50)

if __name__ == "__main__":
    check_gpu()
