import os
import open3d
import cv2
import csv
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from pathlib import Path

Output = namedtuple("Output", ["rawdatapath", "clusterdatapath", "iou"])

class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        self.truncation = data[1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        
def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Box3D(line) for line in lines if line[0] != 'Dontpedestriane']
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't pedestriane about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]   


def draw_projected_box2d(image, qs, color=(255, 255, 255), thickness=1):
    # qs in the format of [[x,y],[x,y]]
    image = cv2.rectangle(image, (int(qs[0][0]), int(qs[0][1])), (int(qs[1][0]), int(qs[1][1])), color, thickness)
    return image


def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)
    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]
    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]
    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    return img


def render_lidar_predicted_bounding_box_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)
    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]
    bboxCoords = get_box2d_from_coordinates(imgfov_pc_pixel)
    img = draw_projected_box2d(img, bboxCoords)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img


def get_box2d_from_lidar_cluster(pts_velo, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)
    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]
    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]
    bboxCoords = [min(imgfov_pc_pixel[0]), min(imgfov_pc_pixel[1]), max(imgfov_pc_pixel[0]), max(imgfov_pc_pixel[1])]
    return bboxCoords


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def getIoUImage(img, gtCoord, predictedCoord, iou):
    img = cv2.rectangle(img, 
                            (int(gtCoord[0]), int(gtCoord[1])), 
                            (int(gtCoord[2]), int(gtCoord[3])), 
                            (0, 255, 0), 
                            2)
    img = cv2.rectangle(img, 
                            (int(predictedCoord[0]), int(predictedCoord[1])), 
                            (int(predictedCoord[2]), int(predictedCoord[3])), 
                            (0, 0, 255), 
                            2)
    img = cv2.putText(img, "IoU: {:.4f}".format(iou), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    return img


def draw_attack_vs_iou_graph(attackAngleIoUVehicle, attackAngleIoUPedestrian, outputpath):
    plt.plot(list(attackAngleIoUVehicle.keys()), list(attackAngleIoUVehicle.values()), label = "Vehicle")
    plt.plot(list(attackAngleIoUPedestrian.keys()), list(attackAngleIoUPedestrian.values()), label = "Pedestrian")
    plt.xlabel("Attack Angle (degrees)")
    plt.ylabel("IoU (0-1)")
    plt.title("Avg IoU vs Attack angle")
    plt.legend()
    plt.savefig(outputpath)
    plt.show()


def write_contents_to_csv(contents, csv_file):
    with Path(csv_file).open('w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(('Rawdata path', "Clusterdata path", "IoU"))
        writer.writerows([(o.rawdatapath, o.clusterdatapath, o.iou) for o in contents])


def show_gt_boundingboxes(img, labels):
    for label in labels:
        gtCoord = label.box2d
        img = cv2.rectangle(img, 
                            (int(gtCoord[0]), int(gtCoord[1])), 
                            (int(gtCoord[2]), int(gtCoord[3])), 
                            (0, 255, 0), 
                            2)    
    plt.imshow(img)
    plt.show()
    return img


def avg_iou_for_each_attack_angle(data, output):
    attackAngleIoU = {}
    attackAngleCount = {}
    # computing iou for given raw data and detected cluster
    for i in range(len(data)):
        try:
            predictedCoord = get_box2d_from_lidar_cluster(data[i].clusterdata, data[i].calibdata, data[i].imgwidth, data[i].imgheight)
            maxIoU = 0
            for label in data[i].labeldata:
                iou = bb_intersection_over_union(label.box2d, predictedCoord)
                if(iou>maxIoU):
                    maxIoU = iou
        except (ValueError, TypeError):
            # This means the lidar points fall outside the image field of view
            maxIoU = 0
        output[i] = Output(output[i].rawdatapath, output[i].clusterdatapath, maxIoU)
        key = data[i].attackangle
        attackAngleIoU[key] = attackAngleIoU.get(key, 0) + maxIoU
        attackAngleCount[key] = attackAngleCount.get(key, 0) + 1
    # computing avg
    for key in attackAngleIoU:
        attackAngleIoU[key] /= attackAngleCount[key]
    sortedDicIoU = {key:attackAngleIoU[key] for key in sorted(attackAngleIoU.keys())}
    sortedDicCount = {key:attackAngleCount[key] for key in sorted(attackAngleCount.keys())}
    return sortedDicIoU


def show_iou_for_each_attack_angle(lidarData, lidarClustersData, img, labels, calib, attackAngles):
    # assuming only one object in frame
    gtCoord = labels[0].box2d
    img_height, img_width, img_channel = img.shape
    fig,axs = plt.subplots(len(lidarData),3, figsize=(100,100))
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ious = []
    for i in range(len(lidarData)):
        lidarImg = render_lidar_on_image(lidarData[i], np.copy(img), calib, img_width, img_height)
        lidarClusterImg = render_lidar_on_image(lidarClustersData[i], np.copy(img), calib, img_width, img_height)
        predictedCoord = get_box2d_from_lidar_cluster(lidarClustersData[i], calib, img_width, img_height)
        maxIoU = 0
        maxGtCoord = []
        for label in labels:
            gtCoord = label.box2d
            iou = bb_intersection_over_union(gtCoord, predictedCoord)
            if(iou>maxIoU):
                maxIoU = iou
                maxGtCoord = gtCoord
        ious.append(maxIoU)
        iouImage = getIoUImage(np.copy(img), maxGtCoord, predictedCoord, maxIoU)
        axs[i,0].imshow(lidarImg, interpolation='nearest')
        axs[i,0].set_title("Point Cloud over image for attack angle " + str(attackAngles[i]), fontsize=20)
        axs[i,0].set_xticks([])
        axs[i,0].set_yticks([])
        axs[i,1].imshow(lidarClusterImg, interpolation='nearest')
        axs[i,1].set_title("Predicted Clusters over image", fontsize=20)
        axs[i,1].set_xticks([])
        axs[i,1].set_yticks([])
        axs[i,2].imshow(iouImage, interpolation='nearest')
        axs[i,2].set_title("Prediction vs GT Bounding Box with IoU", fontsize=20)
        axs[i,2].set_xticks([])
        axs[i,2].set_yticks([])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.05, hspace=0.05)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-datapath", "--datapath", help="Directory path for the data folder")
    parser.add_argument("-outputpath", "--outputpath", help="Directory path for the output folder")
    args = parser.parse_args()
    path = [os.path.join(args.datapath, "car"), os.path.join(args.datapath, "pedestrian")]
    calibPath = os.path.join(args.datapath, "calib")
    labelPath = os.path.join(args.datapath, "label_2")
    imgPath = os.path.join(args.datapath, "image_2")

    samples = [next(os.walk(path[0]))[1], next(os.walk(path[1]))[1]]
    lidarData = []
    output = []
    attackAngleIoU = [{}, {}]
    Detection = namedtuple("Detection", ["framenum", "attackangle", "rawdata", "clusterdata", "calibdata", "labeldata", "imgdata", "imgwidth", "imgheight"])
    maxAttackAngle = [24,8]
    for i,obstacle in enumerate(samples):
        lidarData.append([])
        output.append([])
        for sample in obstacle:
            calibData = read_calib_file(calibPath + "/" + sample + ".txt")
            labelData = load_label(labelPath + "/" + sample + ".txt")   
            img = cv2.cvtColor(cv2.imread(imgPath + "/" + sample + ".png"), cv2.COLOR_BGR2RGB)
            img_height, img_width, img_channel = img.shape
            for attackAngle in range(maxAttackAngle[i]+1):
                rawData = sample + "_" + str(attackAngle) + ".bin"
                clusterData = sample + "_" + str(attackAngle) + "_clusters_0.bin"
                rawDataPath = os.path.join(path[i], sample, rawData)
                clusterDataPath = os.path.join(path[i], sample, clusterData)
                if(os.path.isfile(rawDataPath) and os.path.isfile(clusterDataPath)):
                    lidarData[-1].append(Detection(sample, attackAngle, load_velo_scan(rawDataPath)[:, :3], load_velo_scan(clusterDataPath)[:, :3], calibData, labelData, img, img_width, img_height))
                    output[-1].append(Output(rawDataPath, clusterDataPath, 0))
                else:
                    lidarData[-1].append(Detection(0,attackAngle,0,0,0,0,0,0,0))
                    output[-1].append(Output(rawDataPath, clusterDataPath, 0))
    
    attackAngleIoU[0] = avg_iou_for_each_attack_angle(lidarData[0], output[0])
    attackAngleIoU[1] = avg_iou_for_each_attack_angle(lidarData[1], output[1])
    write_contents_to_csv(output[0], os.path.join(args.outputpath, "vehicle.csv"))
    write_contents_to_csv(output[1], os.path.join(args.outputpath, "pedestrian.csv"))
    draw_attack_vs_iou_graph(attackAngleIoU[0], attackAngleIoU[1], os.path.join(args.outputpath, "avgIoUvsAttackAngle.png"))
    # shows variations for one sample
    show_iou_for_each_attack_angle([x.rawdata for x in lidarData[1]][:3], [x.clusterdata for x in lidarData[1]][:3], lidarData[1][0].imgdata, lidarData[1][0].labeldata, lidarData[1][0].calibdata, [x.attackangle for x in lidarData[1]][:3])
    # show_gt_boundingboxes(lidarData[0][0].imgdata, lidarData[0][0].labeldata)
