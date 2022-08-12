#!/usr/bin/env python

import numpy as np
import math
from numba import jit
import libfmp.c4
import json
from threading import Lock
import argparse
import warnings


class AudioThumbnailer:
    """Audio thumbnailing based on the dynamic-programming algorithm of Müller et al. (2013) and Jiang et al. (2014).
    """

    def __init__(self, audio_filename, thumbnail_duration_sec=30, thumbnail_search_origin_sec=0,
                 thumbnail_search_step_sec=5, strategy='relative', threshold=0.15, penalty=-2, tempo_num=5,
                 tempo_rel_min=0.66, tempo_rel_max=1.5, downsampling_filter_length=21,
                 smoothing_filter_downsampling_factor=5, essm_filter_length=12):

        """Class constructor; loads an audio file and computes a normalized self-similarity matrix based on
        given model parameters. For details on alternative thresholding strategies and parameters,
        see the FMP documentation.

        Args:
            audio_filename (string): Name of audio file to be thumbnailed
            thumbnail_duration_sec: Nominal thumbnail duration, in seconds (default: 30)
            thumbnail_search_origin_sec: Starting time-point for thumbnail search, in seconds (default: 0)
            thumbnail_search_step_sec: Thumbnail search granularity, in seconds (default 5)

            strategy: Thresholding strategy for the SSM computation ('absolute', 'relative' [default], or 'local')
            threshold: Meaning depends on selected strategy; see libfmp docs (default: 0.15)
            penalty: Value to apply to SSM elements below threshold, in 'relative' strategy only (default: -2).
            tempo_num: Number of logarithmically-spaced relative tempi between minimum and maximum (default: 5)
            tempo_rel_min: Minimum tempo ratio between thumbnail instances (default: 0.66)
            tempo_rel_max: Maximum tempo ratio between thumbnail instances (default: 1.50)
            downsampling_filter_length: Smoothing filter length for downsampling of the feature sequence (default: 16)
            smoothing_filter_downsampling_factor: Feature downsampling factor (default: 5)
            essm_filter_length: Smoothing filter length for enhanced similarity matrix computation (default: 12)

        Instance properties initialized:
            self.audio_duration: Duration of audio input
            self.ssm: Normalized self-similarity matrix of input audio
            self.fs_feature: Feature rate
            self.segment_family: Array containing the boundaries of all thumbnail instances (in seconds)
            self.thumbnail: Boundaries of primary thumbnail in seconds
            self.coverage: Normalized total coverage of all thumbnail instances in the audio
            self.audio_filename: [See description of respective arg]
            self.thumbnail_duration_sec: [See description of respective arg]
            self.thumbnail_search_origin_sec: [See description of respective arg]
            self.thumbnail_search_step_sec: [See description of respective arg]
        """

        self.lock = Lock()

        self.audio_filename = audio_filename
        self.thumbnail_duration_sec = thumbnail_duration_sec
        self.thumbnail_search_origin_sec = thumbnail_search_origin_sec
        self.thumbnail_search_step_sec = thumbnail_search_step_sec

        tempo_rel_set = libfmp.c4.compute_tempo_rel_set(tempo_rel_min, tempo_rel_max, tempo_num)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            _, audio_duration, _, fs_feature, ssm, _ = \
                libfmp.c4.compute_sm_from_filename(self.audio_filename,
                                                   L=downsampling_filter_length,
                                                   H=smoothing_filter_downsampling_factor,
                                                   L_smooth=essm_filter_length,
                                                   tempo_rel_set=tempo_rel_set,
                                                   penalty=penalty,
                                                   thresh=threshold,
                                                   strategy=strategy)

        self.properties_ssm = AudioThumbnailer.normalization_properties_ssm(ssm)
        self.ssm = self.properties_ssm
        self.audio_duration = audio_duration
        self.fs_feature = fs_feature
        self.segment_family = None
        self.thumbnail = None
        self.coverage = None

    @staticmethod
    def cli():
        """Adds a command-line interface and parses arguments, displaying help otherwise.

        Returns:
            parser: An ```argparser``` object with command line arguments
        """

        parser = argparse.ArgumentParser(description='Audio file thumbmailing.')
        parser.add_argument('audio_filename', metavar='file', type=str,
                            help='Name of audio file to be thumbnailed')
        parser.add_argument('--thumbnail_duration_sec', '-d', metavar='N', type=int, nargs='?', default=30,
                            help='Name of audio file to be thumbnailed')
        parser.add_argument('--thumbnail_search_origin_sec', '-o', metavar='N', type=int, nargs='?', default=0,
                            help='Starting time-point for thumbnail search, in seconds')
        parser.add_argument('--thumbnail_search_step_sec', '-s', metavar='N', type=int, nargs='?', default=5,
                            help='Thumbnail search granularity, in seconds')
        parser.add_argument('--strategy', '-r', metavar='strategy', type=str, nargs='?', default='relative',
                            choices=['absolute', 'relative', 'local'],
                            help="Thresholding strategy for SSM computation")
        parser.add_argument('--threshold', '-t', metavar='X', type=float, nargs='?', default=0.15,
                            help='Meaning depends on selected strategy; see libfmp docs')
        parser.add_argument('--penalty', '-p', metavar='N', type=int, nargs='?', default=-2,
                            help="Value to apply to SSM elements below threshold, in 'relative' strategy only")
        parser.add_argument('--tempo_num', '-n', metavar='N', type=int, nargs='?', default=5,
                            help='Number of logarithmically-spaced relative tempi between minimum and maximum')
        parser.add_argument('--tempo_rel_min', '-m', metavar='X', type=float, nargs='?', default=0.66,
                            help='Minimum tempo ratio between thumbnail instances')
        parser.add_argument('--tempo_rel_max', '-M', metavar='X', type=float, nargs='?', default=1.50,
                            help='Maximum tempo ratio between thumbnail instances')
        parser.add_argument('--downsampling_filter_length', '-f', metavar='N', type=int, nargs='?', default=16,
                            help='Smoothing filter length for downsampling of the feature sequence')
        parser.add_argument('--smoothing_filter_downsampling_factor', '-i', metavar='N', type=int, nargs='?', default=5,
                            help='Feature downsampling factor')
        parser.add_argument('--essm_filter_length', '-F', metavar='N', type=int, nargs='?', default=12,
                            help='Smoothing filter length for enhanced similarity matrix computation')
        parser.parse_args()

        # convert the NameSpace object returned by the ArgumentParser to an unpackable Python dictionary
        args = vars(parser.parse_args())

        return args

    def run(self):
        """Locates primary thumbnail and any additional instances thereof, wrapping them within a JSON object.

        Args:

        Instance properties set:
            self.thumbnail: Boundaries of primary thumbnail in seconds
        """

        self.lock.acquire()

        max_fitness = 0
        index = 0
        while index in np.arange(self.thumbnail_search_origin_sec,
                                 self.audio_duration - self.thumbnail_duration_sec,
                                 self.thumbnail_search_step_sec):

            # boundaries of current thumbnail candidate in seconds
            seg_sec = [
                index,
                index + self.thumbnail_duration_sec
            ]

            # boundaries of current thumbnail candidate in feature space
            seg = [
                int(seg_sec[0] * self.fs_feature),
                int(seg_sec[1] * self.fs_feature)
            ]

            # self-similarity sub-matrix for current thumbnail candidate
            s_seg = self.ssm[:, seg[0]:seg[1] + 1]

            # accumulated score matrix and score of current thumbnail candidate
            d, score = AudioThumbnailer.compute_accumulated_score_matrix(s_seg)

            # optimal path family (i.e. thumbnail instance family) for current candidate
            path_family = AudioThumbnailer.compute_optimal_path_family(d)

            # boundaries of optimal path family members
            n = self.ssm.shape[0]
            segment_family, coverage = AudioThumbnailer.compute_induced_segment_family_coverage(path_family)

            # quality metrics of current thumbnail candidate
            fitness, score, score_n, coverage, coverage_n, path_family_length = \
                AudioThumbnailer.compute_fitness(path_family, score, n)

            if fitness > max_fitness:

                max_fitness = fitness

                self.thumbnail = {
                    "filename": self.audio_filename,
                    "thumbnail": {
                        "boundaries_in_seconds": json.dumps((segment_family / self.fs_feature).tolist()),
                        "fitness": '%0.3f' % fitness,
                        "nominal_duration_in_seconds": self.thumbnail_duration_sec,
                        "search_step_in_seconds": self.thumbnail_search_step_sec,
                        "total_coverage_in_seconds": coverage / self.fs_feature,
                        "normalized_total_coverage": '%0.3f' % coverage_n,
                        "score": '%0.3f' % score,
                        "normalized_score": '%0.3f' % score_n
                    },
                    "context": {
                        "audio_duration_seconds": '{:.2f}'.format(self.audio_duration),
                        "feature_rate_hz": self.fs_feature,
                        "ssm_dimensions": {
                            "x": self.ssm.shape[0],
                            "y": self.ssm.shape[1]
                        }
                    }
                }

            index += self.thumbnail_search_step_sec

        self.lock.release()

        print(json.dumps(self.thumbnail, indent=2))

    @staticmethod
    def normalization_properties_ssm(ssm):
        """Normalizes self-similarity matrix to fulfill S(n,n)=1, issuing a warning if max(S)<=1 is not fulfilled.

        Args:
            ssm (np.ndarray): Self-similarity matrix (SSM)

        Returns:
            ssm_normalized (np.ndarray): Normalized self-similarity matrix
        """
        ssm_normalized = ssm.copy()
        size = ssm_normalized.shape[0]
        max_s = math.inf
        for n in range(size):
            ssm_normalized[n, n] = 1
            max_s = np.max(ssm_normalized)
        if max_s > 1:
            print('Normalization condition for SSM not fulfill (max > 1)')

        return ssm_normalized

    @staticmethod
    def compute_induced_segment_family_coverage(path_family):
        """ Induces a segment family and computes its absolute coverage given a path family.

        Args:
            path_family (list): Path family

        Returns:
            segment_family (np.ndarray): Induced segment family
            coverage (float): Absolute coverage of path family
        """
        num_path = len(path_family)
        coverage = 0
        if num_path > 0:
            segment_family = np.zeros((num_path, 2), dtype=int)
            for n in range(num_path):
                segment_family[n, 0] = path_family[n][0][0]
                segment_family[n, 1] = path_family[n][-1][0]
                coverage = coverage + segment_family[n, 1] - segment_family[n, 0] + 1
        else:
            segment_family = np.empty

        return segment_family, coverage

    @staticmethod
    @jit(nopython=True)
    def compute_accumulated_score_matrix(S_seg):
        """Computes the accumulated score matrix for the self-similarity sub-matrix corresponding to the timespan of
        a potential thumbnail.

        Args:
            S_seg (np.ndarray): Sub-matrix of an enhanced and normalized SSM ``S``
                Note: ``S`` must satisfy ``S(n,m) <= 1 and S(n,n) = 1``,
                where ``m`` is the duration of a potential thumbnail

        Returns:
            D (np.ndarray): Accumulated score matrix
            score (float): Score of optimal path family
        """
        inf = math.inf
        N = S_seg.shape[0]
        M = S_seg.shape[1] + 1

        # Initializing score matrix
        D = -inf * np.ones((N, M), dtype=np.float64)
        D[0, 0] = 0.
        D[0, 1] = D[0, 0] + S_seg[0, 0]

        # Dynamic programming
        for n in range(1, N):
            D[n, 0] = max(D[n - 1, 0], D[n - 1, -1])
            D[n, 1] = D[n, 0] + S_seg[n, 0]
            for m in range(2, M):
                D[n, m] = S_seg[n, m - 1] + max(D[n - 1, m - 1], D[n - 1, m - 2], D[n - 2, m - 1])

        # Score of optimal path family
        score = np.maximum(D[N - 1, 0], D[N - 1, M - 1])

        return D, score

    @staticmethod
    @jit(nopython=True)
    def compute_optimal_path_family(D):
        """Computes an optimal path family given an accumulated score matrix.

        Args:
            D (np.ndarray): Accumulated score matrix

        Returns:
            path_family (list): Optimal path family consisting of list of paths
                (each path being a list of index pairs)
        """
        # Initialization
        inf = math.inf
        N = int(D.shape[0])
        M = int(D.shape[1])
        cell = (0, 0)

        path_family = []
        path = []

        n = N - 1
        if D[n, M - 1] < D[n, 0]:
            m = 0
        else:
            m = M - 1
            path_point = (N - 1, M - 2)
            path.append(path_point)

        # Backtracking
        while n > 0 or m > 0:

            # obtaining the set of possible predecessors given our current position
            if n <= 2 and m <= 2:
                predecessors = [(n - 1, m - 1)]
            elif n <= 2 and m > 2:
                predecessors = [(n - 1, m - 1), (n - 1, m - 2)]
            elif n > 2 and m <= 2:
                predecessors = [(n - 1, m - 1), (n - 2, m - 1)]
            else:
                predecessors = [(n - 1, m - 1), (n - 2, m - 1), (n - 1, m - 2)]

            # case for the first row. Only horizontal movements allowed
            if n == 0:
                cell = (0, m - 1)
            # case for the elevator column: we can keep going down the column or jumping to the end of the next row
            elif m == 0:
                if D[n - 1, M - 1] > D[n - 1, 0]:
                    cell = (n - 1, M - 1)
                    path_point = (n - 1, M - 2)
                    if len(path) > 0:
                        path.reverse()
                        path_family.append(path)
                    path = [path_point]
                else:
                    cell = (n - 1, 0)
            # case for m=1, only horizontal steps to the elevator column are allowed
            elif m == 1:
                cell = (n, 0)
            # regular case
            else:

                # obtaining the best of the possible predecessors
                max_val = -inf
                for i, cur_predecessor in enumerate(predecessors):
                    if max_val < D[cur_predecessor[0], cur_predecessor[1]]:
                        max_val = D[cur_predecessor[0], cur_predecessor[1]]
                        cell = cur_predecessor

                # saving the point in the current path
                path_point = (cell[0], cell[1] - 1)
                path.append(path_point)

            (n, m) = cell

        # adding last path to the path family
        path.reverse()
        path_family.append(path)
        path_family.reverse()

        return path_family

    @staticmethod
    def compute_fitness(path_family, score, N):
        """Computes the fitness measure and other metrics of a path family.

        Args:
            path_family (list): Path family
            score (float): Score
            N (int): Length of feature sequence

        Returns:
            fitness (float): Fitness
            score (float): Score
            score_n (float): Normalized score
            coverage (float): Coverage
            coverage_n (float): Normalized coverage
            path_family_length (int): Length of path family (total number of cells)
        """
        eps = 1e-16
        num_path = len(path_family)
        M = path_family[0][-1][1] + 1

        # Normalized score
        path_family_length = 0
        for n in range(num_path):
            path_family_length = path_family_length + len(path_family[n])
        score_n = (score - M) / (path_family_length + eps)

        # Normalized coverage
        segment_family, coverage = AudioThumbnailer.compute_induced_segment_family_coverage(path_family)
        coverage_n = (coverage - M) / (N + eps)

        # Fitness measure
        fitness = 2 * score_n * coverage_n / (score_n + coverage_n + eps)

        return fitness, score, score_n, coverage, coverage_n, path_family_length


if __name__ == "__main__":
    args = AudioThumbnailer.cli()
    t = AudioThumbnailer(**args)
    t.run()
