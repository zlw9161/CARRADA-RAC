"""Function to create JSON databases"""
import os
import glob
import json
import time
import numpy as np

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils.transform_annotations import AnnotationTransformer
from carrada_dataset.utils.generate_annotations import AnnotationGenerator

import matplotlib.pyplot as plt

from carrada_dataset.utils.visualize_signal import SignalVisualizer

RA_SHAPE = (256, 256)
RD_SHAPE = (256, 64)


def get_instance_oriented(sequences, instance_exceptions, carrada_path, write_results=True):
    """
    Function to generate annotation file oriented by instance
    For each sequence, the keys are the observed instances.
    For each instance, keys are the frames in which it appears (with annotations)

    PARAMETERS
    ----------
    sequences: list of str
        Names of the sequences to process
    instance exceptions: dict
        Manage instance merging
    carrada_path: str
        Path to Carrada dataset

    RETURNS
    -------
    annotations: dict
        Formated annotations
    """

    with open(os.path.join(carrada_path, 'data_seq_ref.json'), 'r') as fp:
        data_seq_ref = json.load(fp)
    save_path = os.path.join(carrada_path, 'new_annotations_instance_oriented.json')
    annotations = dict()
    for sequence in sequences:
        print('***** Processing sequence: {} *****'.format(sequence))
        annotations[sequence] = dict()
        raw_annotations_paths = os.path.join(carrada_path, sequence,
                                             'centroid_tracking_cluster_jensen_shannon.json')
        rd_annotations_paths = os.path.join(carrada_path, sequence,
                                            'clb_rd_points_cluster.json')
        ra_annotations_paths = os.path.join(carrada_path, sequence,
                                            'clb_ra_points_second.json')
        try:
            with open(raw_annotations_paths, 'r') as fp:
                raw_annotations = json.load(fp)
        except FileNotFoundError:
            print('Annotations have not been generated for sequence: {}.'.format(sequence))
            continue
        for instance in raw_annotations.keys():
            if sequence in instance_exceptions.keys() and \
                    instance in instance_exceptions[sequence].keys():
                clean_instance = instance_exceptions[sequence][instance]
            else:
                clean_instance = instance

            if clean_instance not in annotations[sequence].keys():
                annotations[sequence][clean_instance] = dict()

            label_index = data_seq_ref[sequence]['instances'].index(instance)
            label = data_seq_ref[sequence]['labels'][label_index]

            all_frames = list(raw_annotations[instance].keys())
            all_frames.sort()
            all_frames = all_frames[1:-1]
            for frame in all_frames:
                annotations[sequence][clean_instance][frame] = dict()
                raw_points = raw_annotations[instance][frame]
                for signal_type in ['range_doppler', 'range_angle']:
                    annotations[sequence][clean_instance][frame][signal_type] = dict()
                    if signal_type == 'range_doppler':
                        data_size = [256, 64]
                        # range = rd_points[frame][instance]['range']
                        # doppler = rd_points[frame][instance]['doppler']
                        # rd_single_point = [range, doppler]
                        annots = AnnotationTransformer(raw_points, RD_SHAPE)
                        # rd_annot_points = get_annot_points(data_size, rd_single_point, annots.to_rd())

                        annot_generator = AnnotationGenerator(RD_SHAPE, annots.to_rd())
                    elif signal_type == 'range_angle':
                        data_size = [256,256]
                        # range = ra_points[frame][instance]['range']
                        # angle = ra_points[frame][instance]['angle']
                        # ra_single_point = [range,angle]
                        annots = AnnotationTransformer(raw_points, RA_SHAPE)
                        # ra_annot_points = get_annot_points(data_size, ra_single_point, annots.to_ra())

                        annot_generator = AnnotationGenerator(RA_SHAPE, annots.to_ra())
                    else:
                        raise TypeError('Signal type {} not supported'.format(signal_type))
                    points = annot_generator.get_points().tolist()
                    box = annot_generator.get_box()
                    box = [[int(coord[0]), int(coord[1])] for coord in box]
                    mask = annot_generator.get_mask()
                    mask_coords = np.where(mask == True)
                    mask = [[int(x), int(y)] for x, y in zip(mask_coords[0], mask_coords[1])]
                    annotations[sequence][clean_instance][frame][signal_type]['sparse'] = points
                    annotations[sequence][clean_instance][frame][signal_type]['box'] = box
                    annotations[sequence][clean_instance][frame][signal_type]['dense'] = mask
                    annotations[sequence][clean_instance][frame][signal_type]['label'] = label
    if write_results:
        with open(save_path, 'w') as fp:
            json.dump(annotations, fp)
    return annotations


def get_frame_oriented(sequences, instance_exceptions, carrada_path, write_results=True):
    """
    Function to generate annotation file oriented by frame.
    For each sequence, each frame has a dict.
    If the frame has annotations, it will have an instance key with all the annotations.

    PARAMETERS
    ----------
    sequences: list of str
        Names of the sequences to process
    instance exceptions: dict
        Manage instance merging
    carrada_path: str
        Path to Carrada dataset

    RETURNS
    -------
    annotations: dict
        Formated annotations
    """

    with open(os.path.join(carrada_path, 'data_seq_ref.json'), 'r') as fp:
        data_seq_ref = json.load(fp)
    save_path = os.path.join(carrada_path, 'new_annotations_frame_oriented.json')
    annotations = dict()
    for sequence in sequences:
        print('*** Processing sequence: {} ***'.format(sequence))
        annotations[sequence] = dict()
        box_light_rd = dict()
        box_light_ra = dict()
        rd_matrix_paths = os.path.join(carrada_path, sequence, 'range_doppler_numpy')
        ra_matrix_paths = os.path.join(carrada_path, sequence, 'range_angle_numpy')
        rd_paths = glob.glob(os.path.join(rd_matrix_paths, '*.npy'))
        rd_paths.sort()
        ra_paths = glob.glob(os.path.join(ra_matrix_paths, '*.npy'))
        ra_paths.sort()
        raw_annotations_paths = os.path.join(carrada_path, sequence,
                                             'centroid_tracking_cluster_jensen_shannon.json')
        ra_annotations_paths = os.path.join(carrada_path, sequence,
                                            'new_ra_points.json')
        sparse_annotations_paths = os.path.join(carrada_path, sequence, 'new_annotations', 'sparse')
        if not os.path.exists(sparse_annotations_paths):
            os.makedirs(sparse_annotations_paths)
        dense_annotations_paths = os.path.join(carrada_path, sequence, 'new_annotations', 'dense')
        if not os.path.exists(dense_annotations_paths):
            os.makedirs(dense_annotations_paths)
        box_annotations_paths = os.path.join(carrada_path, sequence, 'new_annotations', 'box')
        if not os.path.exists(box_annotations_paths):
            os.makedirs(box_annotations_paths)
        image_annotations_paths = os.path.join(carrada_path, sequence, 'new_annotations', 'image')
        if not os.path.exists(image_annotations_paths):
            os.makedirs(image_annotations_paths)

        try:
            with open(raw_annotations_paths, 'r') as fp:
                raw_annotations = json.load(fp)
            with open(ra_annotations_paths, 'r') as fp:
                ra_annotations = json.load(fp)
        except FileNotFoundError:
            print('Annotations have not been generated for sequence: {}.'.format(sequence))
            continue

        frame_ids = glob.glob(os.path.join(carrada_path, sequence, 'range_doppler_numpy',
                                           '*.npy'))
        frame_ids.sort()
        # zlw@20211009
        frames_ids = [frame.split('/')[-1].split('.')[0] for frame in frame_ids]
        for frame in frames_ids:
            annotations[sequence][frame] = dict()
            sparse_rd_data_background = np.ones([256, 64])
            sparse_rd_data = np.zeros([4, 256, 64])
            sparse_rd_data[0] = sparse_rd_data_background

            dense_rd_data_background = np.ones([256, 64])
            dense_rd_data = np.zeros([4, 256, 64])
            dense_rd_data[0] = dense_rd_data_background

            sparse_ra_data_background = np.ones([256, 256])
            sparse_ra_data = np.zeros([4, 256, 256])
            sparse_ra_data[0] = sparse_ra_data_background

            dense_ra_data_background = np.ones([256, 256])
            dense_ra_data = np.zeros([4, 256, 256])
            dense_ra_data[0] = dense_ra_data_background

            sparse_paths = os.path.join(sparse_annotations_paths, frame)
            if not os.path.exists(sparse_paths):  # sparse annot saving path check
                os.makedirs(sparse_paths)  # create saving path

            dense_paths = os.path.join(dense_annotations_paths, frame)
            if not os.path.exists(dense_paths):  # dense annot saving path check
                os.makedirs(dense_paths)  # create saving path

            for instance in raw_annotations.keys():
                label_index = data_seq_ref[sequence]['instances'].index(instance)
                label = data_seq_ref[sequence]['labels'][label_index]
                all_frames = list(raw_annotations[instance].keys())
                all_frames.sort()
                all_frames = all_frames[1:-1]
                if frame in all_frames:
                    box_light_ra[frame] = dict()
                    box_light_rd[frame] = dict()
                    raw_points = raw_annotations[instance][frame]
                    if sequence in instance_exceptions.keys() and \
                            instance in instance_exceptions[sequence].keys():
                        clean_instance = instance_exceptions[sequence][instance]
                    else:
                        clean_instance = instance
                    annotations[sequence][frame][clean_instance] = dict()
                    for signal_type in ['range_doppler', 'range_angle']:
                        annotations[sequence][frame][clean_instance][signal_type] = dict()
                        if signal_type == 'range_doppler':
                            data_size = [256, 64]
                            annots = AnnotationTransformer(raw_points, RD_SHAPE)
                            annot_generator = AnnotationGenerator(RD_SHAPE, annots.to_rd())
                            rd_matrix = np.load(rd_paths[int(frame)])
                            os.makedirs(os.path.join(image_annotations_paths, signal_type), exist_ok=True)
                            path_vis_rd = os.path.join(image_annotations_paths, signal_type, frame + '.png')
                            visualise_points_on_rd(rd_matrix, path_vis_rd, annots.to_rd())
                        elif signal_type == 'range_angle':
                            data_size = [256, 256]
                            ra_points = ra_annotations[instance][frame]
                            annot_generator = AnnotationGenerator(RA_SHAPE, ra_points)
                            ra_matrix = np.load(ra_paths[int(frame)])
                            os.makedirs(os.path.join(image_annotations_paths, signal_type), exist_ok=True)
                            path_vis_ra = os.path.join(image_annotations_paths, signal_type, frame + '.png')
                            visualise_points_on_rd(ra_matrix, path_vis_ra, ra_points)
                        else:
                            raise TypeError('Signal type {} not supported'.format(signal_type))
                        points = annot_generator.get_points().tolist()
                        box = annot_generator.get_box()
                        box = [[int(coord[0]), int(coord[1])] for coord in box]
                        mask = annot_generator.get_mask()
                        mask_coords = np.where(mask == True)
                        mask = [[int(x), int(y)] for x, y in zip(mask_coords[0], mask_coords[1])]
                        annotations[sequence][frame][clean_instance][signal_type]['sparse'] = points
                        annotations[sequence][frame][clean_instance][signal_type]['box'] = box
                        annotations[sequence][frame][clean_instance][signal_type]['dense'] = mask
                        annotations[sequence][frame][clean_instance][signal_type]['label'] = label
                        if signal_type == 'range_doppler':
                            box_light_rd[frame]['boxes'] = [box[0][0],box[0][1],box[1][0],box[1][1]]
                            box_light_rd[frame]['labels'] = [label]
                            sparse_rd_data = get_sparse(sparse_rd_data, label, points)
                            dense_rd_data = get_dense(dense_rd_data, label, mask)
                        else:
                            box_light_ra[frame]['boxes'] = [box[0][0], box[0][1], box[1][0], box[1][1]]
                            box_light_ra[frame]['labels'] = [label]
                            sparse_ra_data = get_sparse(sparse_ra_data, label, points)
                            dense_ra_data = get_dense(dense_ra_data, label, mask)
            save_sparse_rd_path = os.path.join(sparse_paths, 'range_doppler.npy')
            save_sparse_ra_path = os.path.join(sparse_paths, 'range_angle.npy')
            np.save(save_sparse_rd_path, sparse_rd_data)
            np.save(save_sparse_ra_path, sparse_ra_data)

            save_dense_rd_path = os.path.join(dense_paths, 'range_doppler.npy')
            save_dense_ra_path = os.path.join(dense_paths, 'range_angle.npy')
            np.save(save_dense_rd_path, dense_rd_data)
            np.save(save_dense_ra_path, dense_ra_data)

        for signal_type in ['range_doppler', 'range_angle']:
            if signal_type == 'range_doppler':
                _ = get_box(box_annotations_paths, signal_type, box_light_rd)
            else:
                _ = get_box(box_annotations_paths, signal_type, box_light_ra)

    if write_results:
        with open(save_path, 'w') as fp:
            json.dump(annotations, fp)
    return annotations


# moving cluster (raw_points) using single_point as new centroid
def get_annot_points(data_size, single_point, raw_points):  
    points_list = list()
    mean_point = np.mean(raw_points, axis=0)
    for point in raw_points:
        difference = [x - y for x, y in zip(point, mean_point)]
        points = [x + y for x, y in zip(single_point, difference)]
        points = [int(points[0]),int(points[1])]
        if points[0] > data_size[0] - 1:
            points[0] = data_size[0] - 1
        if points[1] > data_size[1] - 1:
            points[1] = data_size[1] - 1
        points_list.append(points)

    return points_list


def get_box(box_annotations_paths, signal_type, box_light):
    box_paths = box_annotations_paths

    if not os.path.exists(box_paths): 
        os.makedirs(box_paths)
    if signal_type == 'range_doppler':
        save_path = os.path.join(box_paths,'range_doppler_light.json')
        with open(save_path, 'w') as fp:
            json.dump(box_light, fp)
    else:
        save_path = os.path.join(box_paths, 'range_angle_light.json')
        with open(save_path, 'w') as fp:
            json.dump(box_light, fp)
    return box_paths


def get_sparse(data, label=None, points=None):
    if str(label) == '1':
        for point in points:
            data[1][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    elif str(label) == '2':
        for point in points:
            data[2][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    elif str(label) == '3':
        for point in points:
            data[3][point[0]][point[1]] = 1
            data[0][point[0]][point[1]] = 0
    return data


def get_dense(data, label=None, masks=None):
    if str(label) == '1':
        for mask in masks:
            data[1][mask[0]][mask[1]] = 1
            data[0][mask[0]][mask[1]] = 0
    elif str(label) == '2':
        for mask in masks:
            data[2][mask[0]][mask[1]] = 1
            data[0][mask[0]][mask[1]] = 0
    elif str(label) == '3':
        for mask in masks:
            data[3][mask[0]][mask[1]] = 1
            data[0][mask[0]][mask[1]] = 0
    return data


def visualise_points_on_rd(rd_matrix, path, points):
    """Visualise and record range-Doppler matrices with projected points

    PARAMETERS
    ----------
    rd_matrix: numpy array
        Range-Doppler signal matrix
    path: str
        Path to save the visualisation
    points: numpy array
        Points to visualise in the range-Doppler
    """
    rd_img = SignalVisualizer(rd_matrix).get_image
    for point in points:
        range_coord = point[0]
        doppler_coord = point[1]
        rd_img[range_coord*4:(range_coord*4+4),
               doppler_coord*4:(doppler_coord*4+4)] = [0.,1., 0.]
    plt.imsave(path, rd_img)
    plt.close()


def visualise_points_on_ra(ra_matrix, path, points, range_res, angle_res):
    """Visualise and record range-Angle matrices with projected points

    PARAMETERS
    ----------
    ra_matrix: numpy array
        Range-Doppler signal matrix
    path: str
        Path to save the visualisation
    points: numpy array
        Points to visualise in the range-Doppler
    range_res: float
        Range resolution
    angle_res: float
        Doppler resolution
    """
    ra_img = SignalVisualizer(ra_matrix).get_image
    for point in points:
        range_coord = point[0]
        angle_coord = point[1]
        ra_img[range_coord*4:(range_coord*4+4),
               angle_coord*4:(angle_coord*4+4)] = [0., 0., 0.]
    plt.imsave(path, ra_img)
    plt.close()



def main():
    print('***** Step 4/4: Generate Annotation Files *****')
    time1 = time.time()
    config_path = os.path.join(CARRADA_HOME, 'config.ini')
    config = Configurable(config_path).config
    warehouse = config['data']['warehouse']
    carrada = os.path.join(warehouse, 'Carrada')
    with open(os.path.join(carrada, 'validated_seqs.txt')) as fp:
        sequences = fp.readlines()
    with open(os.path.join(carrada, 'instance_exceptions.json'), 'r') as fp:
        instance_exceptions = json.load(fp)
    sequences = [seq.replace('\n', '') for seq in sequences]
    # annotations_io = get_instance_oriented(sequences, instance_exceptions, carrada)
    annotations_fo = get_frame_oriented(sequences, instance_exceptions, carrada)
    print('***** Execution Time for Step 4/4:'
          ' {} secs. *****'.format(round(time.time() - time1, 2)))


if __name__ == '__main__':
    main()
