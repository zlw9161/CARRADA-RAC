import numpy as np
import os
import json
import glob

from carrada_dataset.annotation_generators.ra_utils import gen_ra_proposals_cfar
from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils.transform_annotations import AnnotationTransformer

RA_SHAPE = (256, 256)

def get_ra_annotations(sequences, carrada_path):
    for sequence in sequences:
        print('*** Processing sequence: {} ***'.format(sequence))
        ra_matrix_paths = os.path.join(carrada_path, sequence, 'range_angle_numpy')
        ra_paths = glob.glob(os.path.join(ra_matrix_paths, '*.npy'))
        ra_paths.sort()

        raw_annotations_paths = os.path.join(carrada_path, sequence,
                                             'centroid_tracking_cluster_jensen_shannon.json')
        new_ra_paths = os.path.join(carrada_path, sequence,
                                    'new_ra_points.json')
        with open(raw_annotations_paths, 'r') as fp:
            raw_annotations = json.load(fp)
        frame_ids = glob.glob(os.path.join(carrada_path, sequence, 'range_angle_numpy',
                                           '*.npy'))
        frame_ids.sort()
        frames_ids = [frame.split('/')[-1].split('.')[0] for frame in frame_ids]

        new_ra_annotations = dict()
        for instance in raw_annotations.keys():
            new_ra_annotations[instance] = dict()
        for frame in frames_ids:
            if int(frame) % 10 == 0:
                print('Processing step: {}/{}'.format(frame, frames_ids[-1]))
            ra_matrix = np.load(ra_paths[int(frame)])
            for instance in raw_annotations.keys():
                all_frames = list(raw_annotations[instance].keys())
                all_frames.sort()
                all_frames = all_frames[1:-1]
                if frame in all_frames:
                    raw_points = raw_annotations[instance][frame]
                    annots = AnnotationTransformer(raw_points, RA_SHAPE)
                    # print(frame, annots.to_ra())
                    new_ra_points = filter_ra(ra_matrix, annots.to_ra())
                    # print(new_ra_points)
                    new_ra_annotations[instance][frame] = new_ra_points
        # print(new_ra_annotations)
        with open(new_ra_paths, 'w') as fp:
            json.dump(new_ra_annotations, fp)
    return


def filter_ra(ra_matrix, ra_points):
    upper_bound = 0
    lower_bound = 255
    for point in ra_points:
        if point[0] > upper_bound:
            upper_bound = point[0]
        if point[0] < lower_bound:
            lower_bound = point[0]
    _, _, ra_matrix_thf = gen_ra_proposals_cfar(ra_matrix)
    new_ra_points = list()
    # print(lower_bound, upper_bound)
    for i in range(lower_bound, upper_bound+1):
        for j in range(ra_matrix.shape[1]):
            if ra_matrix_thf[i][j] > 0:
                new_ra_points.append([i,j])
    # print(new_ra_points)
    if not new_ra_points:
        new_ra_points = ra_points
    return new_ra_points


if __name__ == '__main__':
    config_path = os.path.join(CARRADA_HOME, 'config.ini')
    config = Configurable(config_path).config
    warehouse = config['data']['warehouse']
    carrada = os.path.join(warehouse, 'Carrada')
    with open(os.path.join(carrada, 'validated_seqs.txt')) as fp:
        sequences = fp.readlines()

    sequences = [seq.replace('\n', '') for seq in sequences]
    print()
    get_ra_annotations(sequences, carrada)
