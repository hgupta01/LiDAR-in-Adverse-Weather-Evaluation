import numpy as np
import scipy.spatial

def ROR_fn(cloud:np.ndarray, radius = 0.25, min_neighbour = 3):
    kdtree = scipy.spatial.cKDTree(cloud[:,:3])
    distances, indices = kdtree.query(cloud[:,:3], k=min_neighbour+1, workers=4) # faster than ball query
    return distances[:,min_neighbour]<radius


def SOR_fn(cloud:np.ndarray, num_neighbour=5, std_mul=0.01, distance_upper_bound=np.inf):
    kdtree = scipy.spatial.KDTree(cloud[:,:3])
    distances, indices = kdtree.query(cloud[:,:3], k=num_neighbour+1, distance_upper_bound=distance_upper_bound, workers=4)
    mean_distances = distances[:,1:].sum(axis=1)/num_neighbour
    valid_mean_distances = mean_distances[np.logical_not(np.isinf(mean_distances))] # if distane_upper_bound is not infinite
    
    # Taken from PCL Library
    sum_ = valid_mean_distances.sum()
    sq_sum_ = (valid_mean_distances*valid_mean_distances).sum()
    mean = sum_/len(valid_mean_distances)
    variance = (sq_sum_ - sum_*sum_ / len(valid_mean_distances)) / (len(valid_mean_distances)-1)
    stddev = np.sqrt(variance)
    disance_threshold  = mean + std_mul*stddev
    return mean_distances < disance_threshold


def DROR_fn(cloud:np.ndarray, radius_multiplier = 3, azimuth_angle_deg = 0.7, min_neighbour = 3, min_search_radius=0, max_search_radius=100):
    range_i = np.linalg.norm(cloud[:,:2], axis=1)
    azimuth_angle_rad = azimuth_angle_deg* np.pi / 180
    search_radius_dynamic = radius_multiplier * range_i * azimuth_angle_rad
    
    if min_search_radius>0:
        search_radius_dynamic[search_radius_dynamic<min_search_radius] = min_search_radius
    if max_search_radius<100:
        search_radius_dynamic[search_radius_dynamic>max_search_radius] = max_search_radius
        
    kdtree = scipy.spatial.KDTree(cloud[:,:3])
    distances, indices = kdtree.query(cloud[:,:3], k=min_neighbour+1, workers=4)
    return distances[:,min_neighbour]<search_radius_dynamic


def DSOR_fn(cloud:np.ndarray, num_neighbour=5, std_mul=0.01, range_mul=0.05, distance_upper_bound=np.inf):
    kdtree = scipy.spatial.KDTree(cloud[:,:3])
    distances, indices = kdtree.query(cloud[:,:3], k=num_neighbour+1, distance_upper_bound=distance_upper_bound, workers=4)
    
    mean_distances = distances[:,1:].sum(axis=1)/num_neighbour
    sum_ = mean_distances.sum()
    sq_sum = (mean_distances*mean_distances).sum()
    mean = sum_/len(mean_distances)
    variance = (sq_sum - sum_*sum_ / len(mean_distances)) / (len(mean_distances)-1)
    stddev = np.sqrt(variance)
    disance_threshold  = mean + std_mul*stddev
    
    pnt_range = np.linalg.norm(cloud[:,:3], axis=1)
    dynamic_threshold = disance_threshold * range_mul * pnt_range
    return mean_distances < dynamic_threshold


def LIOR_fn(cloud:np.ndarray, snow_detection_range = 71.2, intensity_th = 5, search_radius=0.1, min_neigbour=3, distance_upper_bound=np.inf):
    assert cloud.shape[-1]>=4
    intensity = cloud[:,3]
    cloud = cloud[:,:3]
    range_i = np.linalg.norm(cloud, axis=1)
    range_inliers = range_i>snow_detection_range # select the points outside the detection range
    intensity_inliers = intensity>intensity_th # select the point which are not snow
    
    # Step1 of finding the intensity 
    inliers = range_inliers | intensity_inliers
    
    # Step 2 Apply Radius Inlier Saving (RIS) to reclassify the outlier as inlier
    kdtree = scipy.spatial.cKDTree(cloud)
    distances, indices = kdtree.query(cloud[np.logical_not(inliers)], k=min_neigbour+1, distance_upper_bound=distance_upper_bound, workers=4)
    outlier_inliers = distances[:,min_neigbour]<search_radius
    # neigbours = kdtree_cloud.query_ball_point(cloud[np.logical_not(inliers)], search_radius, workers=4)
    # num_neigbours = np.asarray([len(x) for x in neigbours])
    # outlier_inliers = num_neigbours>=(min_neigbours+1)
    
    # print("Outlier Inliers: ", outlier_inliers.sum())

    inliers[np.logical_not(inliers)] = outlier_inliers
    # print("Inlier %: ", inliers.sum()/len(cloud))
    return inliers


def distance_2_alphar_map(d):
    if d<10:
        return 0.016
    elif d<20:
        return 0.018
    elif d<30:
        return 0.02
    elif d<40:
        return 0.022
    elif d<50:
        return 0.024
    elif d<60:
        return 0.026
    elif d<70:
        return 0.028
    elif d<80:
        return 0.03
    elif d<90:
        return 0.032
    else:
        return 0.034
    
DIST_BIN = np.array([10,20,30,40,50,60,70,80,90])
ALPHA = np.array([0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.032, 0.034])

def DDIOR_fn(cloud, num_neighbour=5):
    assert cloud.shape[-1]>=4
    intensity = cloud[:,3]
    cloud = cloud[:,:3]
    
    kdtree = scipy.spatial.KDTree(cloud)
    distances, indices = kdtree.query(cloud, k=num_neighbour+1, workers=4)
    
    mean_distances = distances[:,1:].sum(axis=1)/(num_neighbour)
    mean = np.mean(mean_distances)
    
    pnt_range = np.linalg.norm(cloud, axis=1)
    index = np.searchsorted(DIST_BIN, pnt_range, side='left', sorter=None)
    alpha_r = ALPHA[index]
    dynamic_threshold = (alpha_r + 0.1*intensity) * mean * pnt_range
    return mean_distances < dynamic_threshold