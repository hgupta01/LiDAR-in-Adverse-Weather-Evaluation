import glob
import yaml
import time
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from tqdm.auto import tqdm
from tqdm.contrib import tzip
from pointcloud_filters import (ROR_fn, SOR_fn, 
                                DROR_fn, DSOR_fn, 
                                LIOR_fn, DDIOR_fn, AGDOR_fn)

filter_fns = {"ROR": partial(ROR_fn, radius=0.25, min_neighbour=3), 
              "SOR": partial(SOR_fn, num_neighbour=3, std_mul=0.01, distance_upper_bound=np.inf), 
              "DROR": partial(DROR_fn, radius_multiplier=3, azimuth_angle_deg=0.4, min_neighbour=3, min_search_radius=0, max_search_radius=10), 
              "DSOR": partial(DSOR_fn, num_neighbour=3, std_mul=0.01, range_mul=0.05, distance_upper_bound=np.inf), 
              "LIOR": partial(LIOR_fn, snow_detection_range=71.2, intensity_th=1, search_radius=0.1, min_neigbour=3), 
              "DDIOR": partial(DDIOR_fn, num_neighbour=3),
              "AGDOR": partial(AGDOR_fn, num_neighbour=3, multiplier=0.01, intensity_th=9, h_theta_inc=np.deg2rad(0.4))}

def analyze_results(inliers, labels):
    GT_Snow = labels==3
    GT_NonSnow = labels!=3
    GT_Objects = (labels!=3) & (labels!=-1)

    Pred_Snow = np.logical_not(inliers)
    Pred_NonSnow = inliers

    Total_Points = len(labels)

    TP = GT_Snow & Pred_Snow
    TN = GT_NonSnow & Pred_NonSnow
    FP = GT_NonSnow & Pred_Snow
    FN = GT_Snow & Pred_NonSnow
    Pred_Object_As_Snow = GT_Objects & Pred_Snow
    return TP.sum(), TN.sum(), FP.sum(), FN.sum(), Pred_Object_As_Snow.sum(), GT_Objects.sum(), Total_Points


def run_filtering(filter_fn, points):
    start = time.perf_counter()
    inliers = filter_fn(points)
    end = time.perf_counter()
    return inliers, end-start


lidar_files = glob.glob("/media/onepiece/T7/Dataset/WADS/WADS_Data/*/velodyne/*.bin")
label_files = [f.replace('velodyne', 'labels').replace('bin', 'label') for f in lidar_files]

with open("/media/onepiece/T7/Dataset/WADS/code/semantic_kitti.yaml", "r") as f:
    #unlabled/outlier: -1 , vehicle: 0, bicycle: 1, person: 2, falling snow: 3
    semantic_labels = yaml.safe_load(f)["semantic2object"] 
    
index = ['_'.join(f.rsplit('.', 1)[0].split('/')[-3:]).replace('_velodyne', '') for f in lidar_files]

for filter_name in tqdm(filter_fns):
    filter_fn = filter_fns[filter_name]
    df = pd.DataFrame([], index=index,  columns=['Total_Points', 'Total_Object_Points', 'TP', 'TN', 'FP', 'FN', 'Object_As_Snow', 'Exec_Time', 'Max_Dist_BF', 'Max_Dist_AF'])
    
    for scanf, labelf in tzip(lidar_files, label_files):
        idx = '_'.join(scanf.rsplit('.', 1)[0].split('/')[-3:]).replace('_velodyne', '')
        scan = np.fromfile(scanf, dtype=np.float32).reshape((-1,4))
        scan, idx_unique = np.unique(scan, axis=0, return_index=True)
        labels = np.fromfile(labelf, dtype=np.int32)[idx_unique]
        labels = np.asarray([semantic_labels[l] for l in labels])
        
        inliers, exec_time = run_filtering(filter_fn, scan)
        Max_Dist_BF = np.linalg.norm(scan[:,:3], axis=1).max()
        Max_Dist_AF = np.linalg.norm(scan[inliers,:3], axis=1).max()
        
        TP, TN, FP, FN, Pred_Object_As_Snow, Total_Object_Points, Total_Points = analyze_results(inliers, labels)
        df.loc[idx] = Total_Points, Total_Object_Points, TP, TN, FP, FN, Pred_Object_As_Snow, exec_time, Max_Dist_BF, Max_Dist_AF
    
    df.to_pickle(f"results/{filter_name}_res.pkl")


    
    
