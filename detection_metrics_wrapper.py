# Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.

"""Wrapper class for making detection_metrics easier to interpret."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import functools
import logging

import numpy as np
import six

from dlav.metrics.detection.config import constants as const
from dlav.metrics.detection.data.metrics_database import MetricsDatabase
from dlav.metrics.detection.data.types import ClassCategory
from dlav.metrics.detection.data.types import DetectionMetricsException
from dlav.metrics.detection.data.types import EvaluationSettings
from dlav.metrics.detection.devkit import detection_metrics as dm
from dlav.metrics.detection.process import algorithms
from dlav.metrics.detection.process.database_marshaller import add_detection_interest_group
from dlav.metrics.detection.process.database_marshaller import add_difficulty_bucket
from dlav.metrics.detection.process.database_marshaller import add_groundtruth_interest_group
from dlav.metrics.detection.process.database_marshaller import create_primary_views
from dlav.metrics.detection.process.database_marshaller import create_secondary_views
from dlav.metrics.detection.report.detection_results import DetectionResults
from dlav.metrics.detection.utilities.utils import hash_key

log = logging.getLogger(__name__)


class DetectionMetricsWrapper(dm.DetectionMetrics):
    """A wrapper which exposes detection_metrics' internal representations more pythonically."""

    DEFAULT_DATABASE_TYPE = MetricsDatabase
    DEFAULT_RESULTS_TYPE = DetectionResults

    def __init__(
            self, database=None, detections=None, groundtruths=None, images=None,
            configuration=None):
        """
        Construct a DetectionMetrics wrapper.

        Args:
            database: A queryable MetricsDatabase object including tables for groundtruths and
                detections, but optionally images as well.
            configuration: An EvaluationSettings object. If None, the module will attempt to find
                sensible defaults given the provided detections and groundtruths.
            detections: A dictionary of dictionaries of detection objects. Kept for backwards
                compatibility, use only if necessary.
            groundtruth: A dictionary of dictionaries of groundtruth objects. Kept for backwards
                compatibility, use only if necessary.
            images: A collection of image objects. Kept for backwards compatibility, use only if
                necessary.
        """
        # if database does not exist, construct an in-memory database:
        if database is None:
            assert detections is not None and groundtruths is not None, \
                "either a database or a collection of labels must be defined!"
            database = self.DEFAULT_DATABASE_TYPE.create(const.MEMORY_DB_PATH, verbose=False)
        # TODO(@drendleman) - ensure database does not close while the wrapper is instantiated
        database.__enter__()

        # import images, groundtruths detections to this database if the user requests:
        if images is not None:
            database.export_images(images)
        if groundtruths is not None:
            database.export_groundtruths(groundtruths)
        if detections is not None:
            database.export_detections(detections)

        # load bounding boxes, class names, and confidences into the devkit:
        create_primary_views(database)
        devkit_dataset = self._load_dataset(database)
        # try to infer what the configuration might be if the user doesn't give us one:
        configuration = configuration if configuration is not None else \
            self.default_configuration(devkit_dataset)
        devkit_configuration = self._build_configuration(configuration)

        super(DetectionMetricsWrapper, self).__init__(
            dataset=devkit_dataset, configuration=devkit_configuration)

        self.__configuration = configuration
        self.__sql_database = database

        if configuration.recompute_detection_in_path:
            algorithms.standard_detection_in_path(self, configuration)

        if configuration.recompute_detection_weights:
            algorithms.standard_label_weights(self, configuration)

        create_secondary_views(database, configuration)

    @property
    def sql_database(self):
        """SQL database connection used for querying additional information."""
        return self.__sql_database

    @staticmethod
    def _load_dataset(database):
        """Load groundtruth and detection bounding boxes and class names into memory."""
        dataset = dm.DetectionMetricsDataset()
        dataset.appendFrames(
            database._retrieve_rows_iter(const.DEFAULT_QUERY_EXPR % const.IMAGES_EXPR))

        dataset.groundtruths.appendRows(
            database._retrieve_rows_iter(const.DEFAULT_QUERY_EXPR % const.LABELS_GROUNDTRUTH_EXPR))

        dataset.detections.appendRows(
            database._retrieve_rows_iter(const.DEFAULT_QUERY_EXPR % const.LABELS_DETECTION_EXPR))

        return dataset

    @staticmethod
    def default_configuration(dat, dontcare_labels=('dontcare',)):
        """Return the EvaluationSettings object as default configuration.

        A class category will be created for all class names visible in the detection set, matching
        a groundtruth category of the same name (even if one does not exist). An 'unmatched' class
        category will be created if at least one groundtruth has a class that matches no detection's
        class, along with an 'all_unmatched' class_group which will match any detection with no
        corresponding groundtruth to all 'unmatched' groundtruths.

        Args:
            dat: a DetectionMetricsDataset object.
            dontcare_labels: A collection of strings. Defines groundtruth class labels that will
                intentionally remain unmatched, but instead be defined as dontcare.

        Returns:
            An EvaluationSettings object.
        """
        all_det_names = set(dat.detections.class_ids)
        all_gt_names = set(dat.groundtruths.class_ids)

        # all detection classes are grouped into class names:
        matched_categories = all_det_names & all_gt_names
        class_groups = OrderedDict(
            (class_name, ClassCategory(labels=(class_name,)))
            for class_name in sorted(matched_categories)
        )
        # groundtruths which match no detection class are grouped into an unmatched label:
        unmatched_groundtruth_categories = all_gt_names - all_det_names - set(dontcare_labels)
        unmatched_detection_categories = all_det_names - all_gt_names
        if len(unmatched_groundtruth_categories | unmatched_detection_categories) > 0:
            # unmatched detections and unmatched groundtruths are matched together:
            log.warning("unmatched detection categories: %s" % unmatched_detection_categories)
            log.warning("unmatched groundtruth categories: %s" % unmatched_groundtruth_categories)

            class_groups['unmatched'] = ClassCategory(
                detection_labels=tuple(unmatched_detection_categories),
                groundtruth_labels=tuple(unmatched_groundtruth_categories))

        # TODO(drendleman): we should also guess weight_height, image_size, and cvip_area
        return EvaluationSettings(
            class_categories=class_groups, dontcare_classes=dontcare_labels)

    @classmethod
    def _build_configuration(cls, configuration):
        """Build a DetectionMetrics configuration object from an EvaluationSettings object."""
        devkit_configuration = dm.MetricsConfiguration(
            class_groups=cls._build_class_groups(configuration),
            dontcare=cls._build_dontcare_configuration(configuration),
            ignore_detection_difficulty_bucket=configuration.ignore_detection_difficulty,
            ignore_groundtruth_difficulty_bucket=configuration.ignore_groundtruth_difficulty,
        )

        return devkit_configuration

    @staticmethod
    def _build_dontcare_configuration(configuration):
        """Construct a DetectionMetrics Dontcare object from an EvaluationSettings object."""
        return dm.DontcareDefinition(
            classes=set(configuration.dontcare_classes)
        )

    @classmethod
    def _build_class_groups(cls, configuration):
        """Set class definitions from an EvaluationSettings object."""
        class_groups = {}
        for name, definition in six.iteritems(configuration.class_categories):
            definition.iou_overlap = 0.5 # JW
            print(definition.iou_overlap)
            class_group = dm.ClassGroup(
                detection_classes=set(definition.detection_labels),
                groundtruth_classes=set(definition.groundtruth_labels),
                iou_threshold= definition.iou_overlap
            )
            class_groups[name] = class_group
        return class_groups

    def _query_label_columns(self, dtype, query, args, dat):
        """Retrieve SQL rows index by (frame_id, label_id) as an NDRecArray."""
        # TODO(@drendleman) - check that all columns are included
        excluded_columns = 0
        result = np.empty_like(dat.labels, dtype=dtype)
        for row in self.__sql_database._retrieve_rows_iter(query, args):
            # TODO(@drendleman) - you can't slice an sqlite row iterator!
            try:
                label_index = dat.idPairToIndex(row[0], row[1])
            except (KeyError, IndexError):
                # KeyError: either the frame_id or label_id does not exist
                # IndexError: the (frame_id, label_id) combo does not exist
                excluded_columns += 1
                continue
            converted_row = tuple(row[col_name] for col_name, col_type in dtype)
            result[label_index] = converted_row

        if excluded_columns > 0:
            log.warning("Dropped %d rows from label query!" % excluded_columns)
        return result

    def query_image_columns(self, dtype, query, args=()):
        """
        Retrieve SQL rows indexed by frame_id as an NDRecArray.

        Args:
            dtype: dtype of the NDRecArray. Field names must be equal to the columns of the query.
            query: SQL used to query the dataset.
            args: Any arguments passed through to the dataset query.

        Returns:
            An NDRecArray of length num_frames.
        """
        excluded_columns = 0
        frame_ids = self.dataset.frame_ids
        result = np.empty_like(self.dataset.frames, dtype=dtype)
        for row in self.__sql_database._retrieve_rows_iter(query, args):
            try:
                frame_index = frame_ids[row[0]]
            except KeyError:
                excluded_columns += 1
                continue
            converted_row = tuple(row[col_name] for col_name, col_type in dtype)
            result[frame_index] = converted_row

        if excluded_columns > 0:
            log.warning("Dropped %d rows from image query!" % excluded_columns)
        return result

    def query_detection_columns(self, dtype, query, args=()):
        """
        Retrieve SQL rows indexed by (frame_id, detection_id) as an NDRecArray.

        Args:
            dtype: dtype of the NDRecArray. Field names must be equal to the columns of the query.
            query: SQL used to query the dataset.
            args: Any arguments passed through to the dataset query.

        Returns:
            An NDRecArray of length num_detections.
        """
        return self._query_label_columns(dtype, query, args, self.dataset.detections)

    def query_groundtruth_columns(self, dtype, query, args=()):
        """
        Retrieve SQL rows indexed by (frame_id, groundtruth_id) as an NDRecArray.

        Args:
            dtype: dtype of the NDRecArray. Field names must be equal to the columns of the query.
            query: SQL used to query the dataset.
            args: Any arguments passed through to the dataset query.

        Returns:
            An NDRecArray of length num_groundtruths.
        """
        return self._query_label_columns(dtype, query, args, self.dataset.groundtruths)

    def get_bucket_id(self, bucket):
        """Retrieve the identifier of a bucket defined in the SQL dataset."""
        # TODO(@drendleman) Detection bucket as well as Groundtruth bucket
        if self.dataset.hasBucket(bucket):
            return bucket

        gt_buckets, dt_buckets = self.__sql_database.get_difficulty_bucket_ids()
        gt_buckets, dt_buckets = set(gt_buckets), set(dt_buckets)
        all_buckets = gt_buckets | dt_buckets
        if bucket not in all_buckets:
            # test if bucket name is one we've defined:
            raise DetectionMetricsException(
                "Undefined bucket %s, must be one of %s" % (bucket, all_buckets))

        log.debug("Querying difficulty bucket %s..." % bucket)
        gt_query, dt_query = None, None
        if bucket in gt_buckets:
            gt_query = self.__sql_database._retrieve_rows_iter(
                const.DEFAULT_QUERY_EXPR % const.DIFFICULTY_GROUNDTRUTH_EXPR % bucket)
        if bucket in dt_buckets:
            dt_query = self.__sql_database._retrieve_rows_iter(
                const.DEFAULT_QUERY_EXPR % const.DIFFICULTY_DETECTION_EXPR % bucket)

        num_ignored = self.dataset.appendBucket(bucket, gt_query, dt_query)

        if num_ignored > 0:
            log.warning("Ignored %d rows from difficulty bucket %s!" % (num_ignored, bucket))

        return bucket

    def anonymous_bucket_id(self, bucket):
        """Construct an identifier of an anonymous EvaluationBucket or SQL statement."""
        name = hash_key(bucket)

        if self.dataset.groundtruths.hasBucket(name):
            return name

        add_difficulty_bucket(self.__sql_database, name, bucket)
        return self.get_bucket_id(name)

    def anonymous_bucket(self, bucket):
        """Retrieve an NDRecArray describing an anonymous EvaluationBucket or SQL statement."""
        name = self.anonymous_bucket_id(bucket)
        return self.dataset.groundtruths.getBucket(name), self.dataset.detections.getBucket(name)

    def get_bucket(self, bucket):
        """Retrieve an NDRecArray describing a bucket in the SQL database."""
        name = self.get_bucket_id(bucket)
        return self.dataset.groundtruths.getBucket(name), self.dataset.detections.getBucket(name)

    def _get_interest_group_id(self, dat, interest_group, sql_interest_groups_fn, sql_expr):
        """Retrieve the identifier of an interest group in the SQL database."""
        if dat.hasInterestGroup(interest_group):
            return interest_group

        sql_interest_groups = set(sql_interest_groups_fn())
        if interest_group not in sql_interest_groups:
            # test if bucket name is one we've defined:
            raise DetectionMetricsException(
                "Undefined interest group %s, must be one of %s" %
                (interest_group, sql_interest_groups))

        destination = sql_expr % interest_group
        log.debug("Querying interest group '%s'..." % destination)

        query = const.DEFAULT_QUERY_EXPR % destination
        num_ignored = dat.appendInterestGroup(
            interest_group, self.__sql_database._retrieve_rows_iter(query))

        if num_ignored > 0:
            log.warning("Ignored %d rows from interest group %s!" % (num_ignored, interest_group))

        return interest_group

    def get_groundtruth_interest_group_id(self, interest_group):
        """Retrieve the identifier of a groundtruth interest group in the SQL database."""
        return self._get_interest_group_id(
            self.dataset.groundtruths, interest_group,
            self.__sql_database.get_groundtruth_interest_group_ids,
            const.INTEREST_GROUP_GROUNDTRUTH_EXPR
        )

    def get_detection_interest_group_id(self, interest_group):
        """Retrieve the identifier of a detection interest group in the SQL database."""
        return self._get_interest_group_id(
            self.dataset.detections, interest_group,
            self.__sql_database.get_detection_interest_group_ids,
            const.INTEREST_GROUP_DETECTION_EXPR
        )

    def get_groundtruth_interest_group(self, interest_group):
        """Retrieve an NDRecArray describing a groundtruth interest group in the SQL database."""
        name = self.get_groundtruth_interest_group_id(interest_group)
        return self.dataset.groundtruths.getInterestGroup(name)

    def get_detection_interest_group(self, interest_group):
        """Retrieve an NDRecArray describing a detection interest group in the SQL database."""
        name = self.get_detection_interest_group_id(interest_group)
        return self.dataset.detections.getInterestGroup(name)

    def anonymous_detection_interest_group_id(self, definition):
        """Construct an identifier for an anonymous detection interest group."""
        name = hash_key(definition)

        if self.dataset.detections.hasInterestGroup(name):
            return name

        add_detection_interest_group(self.__sql_database, name, definition)
        return name

    def anonymous_groundtruth_interest_group_id(self, definition):
        """Construct an identifier for an anonymous groundtruth interest group."""
        name = hash_key(definition)

        if self.dataset.groundtruths.hasInterestGroup(name):
            return name

        add_groundtruth_interest_group(self.__sql_database, name, definition)
        return name

    def anonymous_detection_interest_group(self, definition):
        """Retrieve an NDRecArray describing a detection interest group in the SQL database."""
        name = self.anonymous_detection_interest_group_id(definition)
        return self.get_detection_interest_group(name)

    def anonymous_groundtruth_interest_group(self, definition):
        """Retrieve an NDRecArray describing a groundtruth interest group in the SQL database."""
        name = self.anonymous_groundtruth_interest_group_id(definition)
        return self.get_groundtruth_interest_group(name)

    @functools.lru_cache(maxsize=10)
    def compute_metrics_multi_threshold(self, bucket_id, classgroup_set_id, thresholds):
        """
        Compute detections and groundtruth statuses for different thresholds, one per class_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.

        Returns:
            A MetricsResults object.
        """
        bucket_id = self.get_bucket_id(bucket_id)

        log.debug("Computing fine-grained metrics across for bucket=%s, labelset=%s" %
                  (bucket_id, classgroup_set_id))
        return self._computeMetricsMultiThreshold(
            thresholds_class=thresholds,
            bucket_id=bucket_id,
            classgroup_set_id=classgroup_set_id,
        )

    def compute_metrics_single_threshold(self, bucket_id, classgroup_set_id, threshold):
        """
        Compute detections and groundtruth statuses with a single threshold for all class_groups.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            threshold: a threshold to apply to all classes.

        Returns:
            A MetricsResults object.
        """
        thresholds = (threshold,) * self.classgroup_sets[classgroup_set_id].count

        return self.compute_metrics_multi_threshold(bucket_id, classgroup_set_id, thresholds)

    @functools.lru_cache(maxsize=1000)
    def aggregate_detection_metrics_multi_threshold(
            self, bucket_id, classgroup_set_id, thresholds, interest_group_id):
        """
        Summarize a collection of detection statuses, filtered by an interest_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of DetMetricsResults of length num_class_groups.
        """
        interest_group_id = self.get_detection_interest_group_id(interest_group_id)
        result = self.compute_metrics_multi_threshold(bucket_id, classgroup_set_id, thresholds)

        summarized_results = result.summarizeDetections(
            interest_group_id, dm.eSumCriteria.BY_CLASS_GROUP)
        return summarized_results

    @functools.lru_cache(maxsize=1000)
    def aggregate_detection_metrics_multi_threshold_individual_class(
            self, bucket_id, classgroup_set_id, thresholds, interest_group_id):
        """
        Summarize a collection of detection statuses by class_name rather than class_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of DetMetricsResults of length num_class_names.
        """
        interest_group_id = self.get_detection_interest_group_id(interest_group_id)
        result = self.compute_metrics_multi_threshold(bucket_id, classgroup_set_id, thresholds)

        summarized_results = result.summarizeDetections(
            interest_group_id, dm.eSumCriteria.BY_CLASS_NAME)
        return summarized_results

    @functools.lru_cache(maxsize=1000)
    def aggregate_groundtruth_metrics_multi_threshold(
            self, bucket_id, classgroup_set_id, thresholds, interest_group):
        """
        Summarize a collection of groundtruth statuses, filtered by an interest_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of GtMetricsResults of length num_class_groups.
        """
        interest_group = self.get_groundtruth_interest_group_id(interest_group)
        result = self.compute_metrics_multi_threshold(bucket_id, classgroup_set_id, thresholds)

        summarized_results = result.summarizeGroundtruths(
            interest_group, dm.eSumCriteria.BY_CLASS_GROUP)
        return summarized_results

    @functools.lru_cache(maxsize=1000)
    def aggregate_groundtruth_metrics_multi_threshold_individual_class(
            self, bucket_id, classgroup_set_id, thresholds, interest_group):
        """
        Summarize a collection of groundtruth statuses by class_name rather than class_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of GtMetricsResults of length num_class_names.
        """
        interest_group = self.get_groundtruth_interest_group_id(interest_group)
        result = self.compute_metrics_multi_threshold(bucket_id, classgroup_set_id, thresholds)

        summarized_results = result.summarizeGroundtruths(
            interest_group, dm.eSumCriteria.BY_CLASS_NAME)
        return summarized_results

    @functools.lru_cache(maxsize=1000)
    def aggregate_metrics_multi_threshold(
            self,
            bucket_id,
            classgroup_set_id,
            thresholds,
            groundtruth_interest_group,
            detection_interest_group
    ):
        """
        Summarize detections and groundtruths together by class_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            groundtruth_interest_group: identifier for an interest group. Filters groundtruths.
            detection_interest_group: identifier for an interest group. Filters detections.

        Returns:
            An NdRecArray of ThresholdMetricsResults of length num_class_groups.
        """
        groundtruth_interest_group = \
            self.get_groundtruth_interest_group_id(groundtruth_interest_group)
        detection_interest_group = self.get_detection_interest_group_id(detection_interest_group)
        result = self.compute_metrics_multi_threshold(bucket_id, classgroup_set_id, thresholds)

        summarized_results = result.summarize(
            groundtruth_interest_group_id=groundtruth_interest_group,
            detection_interest_group_id=detection_interest_group
        )
        return summarized_results

    def aggregate_detection_metrics_single_threshold(
            self, bucket_id, classgroup_set_id, threshold, interest_group_id):
        """
        Summarizes a collection of detection statuses by a single threshold.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of DetMetricsResults of length num_class_groups.
        """
        thresholds = (threshold,) * self.classgroup_sets[classgroup_set_id].count

        return self.aggregate_detection_metrics_multi_threshold(
            bucket_id, classgroup_set_id, thresholds, interest_group_id)

    def aggregate_groundtruth_metrics_single_threshold(
            self, bucket_id, classgroup_set_id, threshold, interest_group):
        """
        Summarizes a collection of groundtruth statuses by a single threshold.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of GtMetricsResults of length num_class_groups.
        """
        thresholds = (threshold,) * self.classgroup_sets[classgroup_set_id].count

        return self.aggregate_groundtruth_metrics_multi_threshold(
            bucket_id, classgroup_set_id, thresholds, interest_group)

    def aggregate_detection_metrics_single_threshold_individual_class(
            self, bucket_id, classgroup_set_id, threshold, interest_group_id):
        """
        Summarizes a collection of detection statuses by class_name rather than class_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of DetMetricsResults of length num_class_groups.
        """
        thresholds = (threshold,) * self.classgroup_sets[classgroup_set_id].count

        return self.aggregate_detection_metrics_multi_threshold_individual_class(
            bucket_id, classgroup_set_id, thresholds, interest_group_id)

    def aggregate_groundtruth_metrics_single_threshold_individual_class(
            self, bucket_id, classgroup_set_id, threshold, interest_group):
        """
        Summarizes a collection of groundtruth statuses by class_name rather than class_group.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            interest_group_id: identifier for an interest group. Filters status aggregation.

        Returns:
            An NdRecArray of GtMetricsResults of length num_class_groups.
        """
        thresholds = (threshold,) * self.classgroup_sets[classgroup_set_id].count

        return self.aggregate_groundtruth_metrics_multi_threshold_individual_class(
            bucket_id, classgroup_set_id, thresholds, interest_group)

    def aggregate_metrics_single_threshold(
            self,
            bucket_id,
            classgroup_set_id,
            threshold,
            groundtruth_interest_group,
            detection_interest_group
    ):
        """
        Summarizes a collection of detection and groundtruth statuses by a single threshold.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.
            thresholds: set of thresholds as a tuple, one threshold per class.
            detection_interest_group: detection interest group identifier.
            detection_interest_group: groundtruth interest group identifier.

        Returns:
            An NdRecArray of GtMetricsResults of length num_class_groups.
        """
        thresholds = (threshold,) * self.classgroup_sets[classgroup_set_id].count

        return self.aggregate_metrics_multi_threshold(
            bucket_id,
            classgroup_set_id,
            thresholds,
            groundtruth_interest_group,
            detection_interest_group
        )

    @functools.lru_cache(maxsize=1000)
    def aggregate_metrics_all_thresholds(self, bucket_id, classgroup_set_id):
        """
        Summarize a collection of MetricsResults objects.

        Used for computing mean-average precision, recall curves, etc.

        Args:
            bucket_id: identifier for a difficulty bucket. Filters label matching.
            classgroup_set_id: identifier for a ClassGroupSet.

        Returns:
            A vector of MetricsResultsAggregateClass objects, length num_class_groups.
        """
        bucket_id = self.get_bucket_id(bucket_id)

        log.info("Computing aggregate metrics across %d thresholds for bucket=%s, labelset=%s" %
                 (self.__configuration.num_metrics_curve_points, bucket_id, classgroup_set_id))
        status = self._computeMetricsAllThresholds(
            num_thresholds=self.__configuration.num_metrics_curve_points,
            difficulty_bucket=bucket_id,
            classgroup_set_id=classgroup_set_id,
            detections_of_interest="total",
            groundtruths_of_interest="total"
        )

        return status.aggregate_class

    def results(self, results_type=None):
        """
        Construct a DetectionResults object of type results_type using this DetectionMetricsWrapper.

        Args:
            results_type (class): A class extending the DetectionResults interface. Defaults to
                DetectionResults.

        Returns:
            An object of class results_type instantiated with the current DetectionMetricsWrapper.
        """
        results_type = results_type if results_type is not None else self.DEFAULT_RESULTS_TYPE

        return results_type(
            metrics=self,
            sql_database=self.__sql_database,
            configuration=self.__configuration)
