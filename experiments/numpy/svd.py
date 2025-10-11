import numpy as np
import scipy as sp
import time

np.show_config()

def measure_svd_time(h,w,svd_func,t=100):
    sum = 0
    for i in range(t):
        A = np.random.rand(h, w)
        start = time.time()
        U, S, Vt = svd_func(A)
        end = time.time()
        sum += end - start
    return sum / t


def compare_svd(h,w,t=100):
    print(f"{h} x {w} matrix SVD comparison over {t} trials:")
    svd_t=measure_svd_time(h,w, lambda A: sp.linalg.svd(A, full_matrices=False, compute_uv=True, lapack_driver='gesvd'),t=t)
    print(f"SVD Thin took {svd_t:.4f} seconds")
    svd_f=measure_svd_time(h,w, lambda A: sp.linalg.svd(A, full_matrices=True, compute_uv=True, lapack_driver='gesvd'),t=t)
    print(f"SVD Full took {svd_f:.4f} seconds")
    sdd_t=measure_svd_time(h,w, lambda A: sp.linalg.svd(A, full_matrices=False, compute_uv=True, lapack_driver='gesdd'),t=t)
    print(f"SVDDC Thin took {sdd_t:.4f} seconds")
    sdd_f=measure_svd_time(h,w, lambda A: sp.linalg.svd(A, full_matrices=True, compute_uv=True, lapack_driver='gesdd'),t=t)
    print(f"SVDDC Full took {sdd_f:.4f} seconds")
    print("")

compare_svd(400, 400)
compare_svd(200, 3000)
