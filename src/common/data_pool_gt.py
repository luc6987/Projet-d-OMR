"""
Ground truth data loader for training assembly notation directly from MUSCIMA++ annotations.
This bypasses YOLO detection and uses ground truth bounding boxes and class labels.
"""
import copy
import os
import logging
from glob import glob
from typing import List, Tuple, Dict
import time

import numpy as np
import yaml
from PIL import Image
from mung.grammar import DependencyGrammar
from mung.graph import NotationGraph
from mung.io import read_nodes_from_file, parse_node_classes
from mung.node import Node

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_grammar(filename):
    """Load dependency grammar for edge validation."""
    mungo_classes_file = os.path.splitext(filename)[0] + '.xml'
    mlclass_dict = {m.name: m for m in parse_node_classes(mungo_classes_file)}
    return DependencyGrammar(grammar_filename=filename, alphabet=set(mlclass_dict.keys()))


def config2data_pool_dict(config):
    """Prepare data pool kwargs from a config dict."""
    data_pool_dict = {
        'max_edge_length': config['THRESHOLD_NEGATIVE_DISTANCE'],
        'max_negative_samples': config['MAX_NEGATIVE_EXAMPLES_PER_OBJECT'],
        'patch_size': (config['PATCH_HEIGHT'], config['PATCH_WIDTH']),
        'zoom': config['IMAGE_ZOOM'],
        'mode': config.get('mode', 'MLP'),
        'filter_pairs': config.get('FILTER_PAIRS', True)  # Default to True if not specified
    }

    if 'PATCH_NO_IMAGE' in config:
        data_pool_dict['patch_no_image'] = config['PATCH_NO_IMAGE']

    # Load grammar, if requested
    if 'RESTRICT_TO_GRAMMAR' in config:
        if not os.path.isfile(config['RESTRICT_TO_GRAMMAR']):
            grammar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        config['RESTRICT_TO_GRAMMAR'])
        else:
            grammar_path = config['RESTRICT_TO_GRAMMAR']

        if os.path.isfile(grammar_path):
            grammar = load_grammar(grammar_path)
            data_pool_dict['grammar'] = grammar
        else:
            logging.warning('Config contains grammar {}, but it is unreachable'
                            ' both as an absolute path and relative to'
                            ' the package. No grammar loaded.'
                            ''.format(config['RESTRICT_TO_GRAMMAR']))

    if "NORMALIZE_BOUNDING_BOXES" in config:
        data_pool_dict['normalize_bbox'] = config['NORMALIZE_BOUNDING_BOXES']
    else:
        data_pool_dict['normalize_bbox'] = False

    return data_pool_dict


class GroundTruthDataPool(Dataset):
    """
    Data pool that uses ground truth annotations from MUSCIMA++ directly.
    This creates pairs of musical notation objects with their bounding boxes and class labels
    for training the assembly/linking model using an MLP.
    """

    def __init__(self, mungs: List[NotationGraph],
                 images: List[np.ndarray],
                 max_edge_length: int,
                 max_negative_samples: int,
                 patch_size: Tuple[int, int],
                 zoom: float,
                 class_list: List[str],
                 class_dict: Dict[str, int],
                 grammar: DependencyGrammar = None,
                 filter_pairs: bool = True,
                 normalize_bbox: bool = True,
                 mode='MLP'):
        """
        Initialize the ground truth data pool.

        :param mungs: NotationGraph objects from ground truth annotations
        :param images: Corresponding images
        :param max_edge_length: Maximum distance between objects to be considered neighbors
        :param max_negative_samples: Maximum negative samples per object
        :param patch_size: Size of extracted patches (height, width)
        :param zoom: Image rescaling factor
        :param class_list: List of all class names to include
        :param class_dict: Dictionary mapping class names to IDs
        :param grammar: Optional grammar for edge validation
        :param filter_pairs: Whether to filter pairs by distance
        :param normalize_bbox: Whether to normalize bounding boxes
        :param mode: Model mode ('MLP', 'MLPwithSoftClass', etc.)
        """
        self.mungs = mungs
        self.images = images

        self.normalize_bbox = normalize_bbox
        self.max_edge_length = max_edge_length
        self.max_negative_samples = max_negative_samples

        self.patch_size = patch_size
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

        self.zoom = zoom
        if self.zoom != 1.0:
            self.__zoom_images()
            self.__zoom_mungs()

        self.class_list = class_list
        self.class_dict = class_dict
        self.grammar = grammar

        self.length = 0
        self.filter_pairs = filter_pairs
        self.mode = mode

        # Prepare training pairs from ground truth
        self.prepare_train_entities()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a single training sample (pair of objects with labels)."""
        mung_from = self.all_mungo_pairs[idx][0]
        mung_to = self.all_mungo_pairs[idx][1]

        # Calculate normalization parameters if needed
        if self.normalize_bbox:
            image_shape = self.images[self.train_entities[idx][0]].shape
            reshape_weight = torch.tensor([2 / image_shape[0], 2 / image_shape[0],
                                           2 / image_shape[0], 2 / image_shape[0]])
            reshape_bias = torch.tensor([-1, -image_shape[1]/image_shape[0],
                                         -1, -image_shape[1]/image_shape[0]])
        else:
            reshape_weight = torch.ones(4)
            reshape_bias = torch.zeros(4)

        # Get bounding boxes and class labels from ground truth
        source_bbox = torch.tensor(mung_from.bounding_box, dtype=torch.float32) * reshape_weight + reshape_bias
        target_bbox = torch.tensor(mung_to.bounding_box, dtype=torch.float32) * reshape_weight + reshape_bias

        # Use ground truth class labels (not predictions)
        source_class = torch.tensor(self.class_dict[mung_from.class_name], dtype=torch.long)
        target_class = torch.tensor(self.class_dict[mung_to.class_name], dtype=torch.long)

        # Ground truth edge label: 1 if objects are linked, 0 otherwise
        label = torch.tensor(
            mung_to.id in mung_from.outlinks or mung_from.id in mung_to.outlinks,
            dtype=torch.float32
        ).unsqueeze(-1)

        return dict(
            node1_id=mung_from.id,
            node2_id=mung_to.id,
            source_bbox=source_bbox,
            source_class=source_class,
            target_bbox=target_bbox,
            target_class=target_class,
            node1_class=source_class,
            node2_class=target_class,
            label=label
        )

    def __zoom_images(self):
        """Apply zoom to all images."""
        images_zoomed = []
        import cv2
        for image in self.images:
            img_copy = image * 1.0
            img_zoomed = cv2.resize(img_copy, dsize=None,
                                    fx=self.zoom, fy=self.zoom).astype(image.dtype)
            images_zoomed.append(img_zoomed)
        self.images = images_zoomed

    def __zoom_mungs(self):
        """Apply zoom to all node bounding boxes."""
        if self.zoom is None or self.zoom == 1.0:
            return
        for mung in self.mungs:
            for m in mung.vertices:
                m.scale(zoom=self.zoom)

    def prepare_train_entities(self):
        """
        Extract object pairs from ground truth for training.
        Creates pairs of objects that are close enough to potentially be linked.
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print("PREPARING TRAINING PAIRS FROM GROUND TRUTH")
        print(f"{'='*60}")
        print(f"Total documents to process: {len(self.mungs)}")
        print(f"Filter pairs by distance: {self.filter_pairs}")
        if self.filter_pairs:
            print(f"Max edge distance: {self.max_edge_length} pixels")
        print(f"{'='*60}\n")

        self.train_entities = []
        self.all_mungo_pairs = []  
        self.inference_graph = {}
        number_of_samples = 0

        doc_times = []

        for mung_index, mung in enumerate(tqdm(self.mungs, desc="Loading ground truth pairs")):
            doc_start = time.time()
            num_nodes = len(mung.vertices)

            if self.filter_pairs:
                # Only consider pairs within max_edge_length distance
                print(f"  [DEBUG] Doc {mung_index+1}/{len(self.mungs)}: {num_nodes} nodes, finding neighbors...", end=" ", flush=True)
                pair_start = time.time()
                total_possible = num_nodes * num_nodes
                object_pairs = self.get_all_neighboring_object_pairs(
                    mung.vertices,
                    max_object_distance=self.max_edge_length,
                    grammar=self.grammar)
                pair_time = time.time() - pair_start
                kept = len(object_pairs)
                discarded = total_possible - kept
                print(f"created {kept} pairs, discarded {discarded} ({discarded/total_possible*100:.1f}%) in {pair_time:.2f}s")
            else:
                # Consider all possible pairs
                print(f"  [DEBUG] Doc {mung_index+1}/{len(self.mungs)}: {num_nodes} nodes, generating all pairs...", end=" ", flush=True)
                pair_start = time.time()
                object_pairs = [(m_from, m_to) for m_from in mung.vertices for m_to in mung.vertices]
                pair_time = time.time() - pair_start
                print(f"created {len(object_pairs)} pairs in {pair_time:.2f}s")

            self.inference_graph[mung_index] = object_pairs

            append_start = time.time()
            for (m_from, m_to) in object_pairs:
                self.all_mungo_pairs.append((m_from, m_to))
                self.train_entities.append([mung_index, number_of_samples])
                number_of_samples += 1
            append_time = time.time() - append_start

            doc_time = time.time() - doc_start
            doc_times.append(doc_time)
            print(f"  [DEBUG] Doc {mung_index+1} total time: {doc_time:.2f}s (append time: {append_time:.2f}s)")

        self.length = number_of_samples
        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"TRAINING PAIRS PREPARED")
        print(f"{'='*60}")
        print(f"Total pairs created: {number_of_samples:,}")
        print(f"Average pairs per document: {number_of_samples/len(self.mungs):.1f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per document: {np.mean(doc_times):.2f}s")
        print(f"{'='*60}\n")

    def get_inference_graph(self):
        """
        Prepare inference graph with all pairs batched as tensors.
        Used during validation/testing.
        """
        start_time = time.time()
        print("[DEBUG] Preparing graph for inference...")
        for idx, graph in tqdm(self.inference_graph.items(), desc="Preparing inference graphs"):
            tensor_dict = {
                "source_bbox": [], "source_class": [],
                "target_bbox": [], "target_class": [],
                "label": [], "source_id": [], "target_id": []
            }

            if self.normalize_bbox:
                image_shape = self.images[idx].shape
                reshape_weight = torch.tensor([2 / image_shape[0], 2 / image_shape[0],
                                            2 / image_shape[0], 2 / image_shape[0]])
                reshape_bias = torch.tensor([-1, -image_shape[1]/image_shape[0],
                                            -1, -image_shape[1]/image_shape[0]])
            else:
                reshape_weight = torch.ones(4)
                reshape_bias = torch.zeros(4)

            for pair in graph:
                # Avoid duplicate pairs
                if pair[0].id > pair[1].id:
                    continue

                source_bbox = torch.tensor(pair[0].bounding_box, dtype=torch.float32) * reshape_weight + reshape_bias
                target_bbox = torch.tensor(pair[1].bounding_box, dtype=torch.float32) * reshape_weight + reshape_bias

                # Ground truth class labels
                source_class = torch.tensor(self.class_dict[pair[0].class_name], dtype=torch.long)
                target_class = torch.tensor(self.class_dict[pair[1].class_name], dtype=torch.long)

                label = torch.tensor(
                    pair[1].id in pair[0].outlinks or pair[0].id in pair[1].outlinks,
                    dtype=torch.float32
                ).unsqueeze(-1)

                tensor_dict["source_bbox"].append(source_bbox)
                tensor_dict["source_class"].append(source_class)
                tensor_dict["target_bbox"].append(target_bbox)
                tensor_dict["target_class"].append(target_class)
                tensor_dict["label"].append(label)
                tensor_dict['source_id'].append(torch.tensor(pair[0].id))
                tensor_dict['target_id'].append(torch.tensor(pair[1].id))

            self.inference_graph[idx] = {k: torch.stack(v) for k, v in tensor_dict.items()}

        total_time = time.time() - start_time
        print(f"[DEBUG] Inference graph preparation completed in {total_time:.2f}s")
        return self.inference_graph

    def get_closest_objects(self, nodes: List[Node], threshold) -> Dict[Node, List[Node]]:
        """Find objects within threshold distance of each node."""
        start_time = time.time()
        close_objects = {}
        for c in nodes:
            close_objects[c] = []

    
        total_comparisons = len(nodes) * len(nodes)
        if total_comparisons > 100000:
            print(f"\n    [DEBUG WARNING: Computing {total_comparisons:,} distance comparisons - this may be slow!]", flush=True)

        distance_calc_start = time.time()
        for c in nodes:
            for d in nodes:
                distance = c.distance_to(d)
                if distance < threshold:
                    close_objects[c].append(d)
                    close_objects[d].append(c)
        distance_calc_time = time.time() - distance_calc_start

        # Remove duplicates
        dedup_start = time.time()
        for key, neighbors in close_objects.items():
            unique_neighbors = list(dict.fromkeys(neighbors))
            close_objects[key] = unique_neighbors
        dedup_time = time.time() - dedup_start

        total_time = time.time() - start_time
        if total_comparisons > 10000:
            print(f"    [DEBUG] Distance calculation: {distance_calc_time:.2f}s, deduplication: {dedup_time:.2f}s, total: {total_time:.2f}s", flush=True)

        return close_objects

    def get_all_neighboring_object_pairs(self, nodes: List[Node],
                                         max_object_distance,
                                         grammar=None) -> List[Tuple[Node, Node]]:
        """Get all pairs of objects within max_object_distance, optionally filtered by grammar."""
        close_neighbors = self.get_closest_objects(nodes, max_object_distance)

       
        total_close_pairs = sum(len(neighbors) for neighbors in close_neighbors.values())

        example_pairs_dict = {}
        grammar_discarded = 0
        for c in close_neighbors:
            if grammar is None:
                example_pairs_dict[c] = close_neighbors[c]
            else:
                before = len(close_neighbors[c])
                example_pairs_dict[c] = [d for d in close_neighbors[c]
                                          if grammar.validate_edge(c.class_name, d.class_name)]
                grammar_discarded += before - len(example_pairs_dict[c])

        examples = []
        for c in example_pairs_dict:
            for d in example_pairs_dict[c]:
                examples.append((c, d))

        
        total_possible_pairs = len(nodes) * len(nodes)
        distance_discarded = total_possible_pairs - total_close_pairs
        final_pairs = len(examples)

        return examples


def load_split(split_file):
    """Load train/val/test split from YAML file."""
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.FullLoader)
    return split


def __load_mung(filename: str, exclude_classes: List[str]) -> NotationGraph:
    """Load ground truth MuNG annotations, excluding specified classes."""
    mungos = read_nodes_from_file(filename)
    mung = NotationGraph(mungos)
    objects_to_exclude = [m for m in mungos if m.class_name in exclude_classes]
    for m in objects_to_exclude:
        mung.remove_vertex(m.id)
    return mung


def __load_image(filename: str) -> np.ndarray:
    """Load image as binary array."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            image = np.array(Image.open(filename).convert('1')).astype('uint8')
            return image
        except OSError as e:
            if attempt < max_retries - 1:
                print(f"    [WARNING] Error loading {filename} (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                time.sleep(0.1)
            else:
                print(f"    [ERROR] Failed to load {filename} after {max_retries} attempts: {e}")
                raise


def __load_ground_truth_data(gt_annotations_root: str, images_root: str,
                             include_names: List[str] = None,
                             max_items: int = None,
                             exclude_classes=None):
    """
    Load ground truth annotations and corresponding images.

    :param gt_annotations_root: Directory containing MUSCIMA++ XML annotation files
    :param images_root: Directory containing image files (png)
    :param include_names: Only load files with these basenames (for splits)
    :param max_items: Maximum number of files to load
    :param exclude_classes: Classes to exclude from loading
    :returns: (mungs, images) tuple of lists
    """
    start_time = time.time()
    if exclude_classes is None:
        exclude_classes = []

    print(f"\n{'='*60}")
    print(f"LOADING GROUND TRUTH DATA")
    print(f"{'='*60}")
    print(f"Annotations root: {gt_annotations_root}")
    print(f"Images root: {images_root}")
    print(f"Number of documents to load: {len(include_names) if include_names else 'all'}")
    if exclude_classes:
        print(f"Excluding {len(exclude_classes)} classes")
    print(f"{'='*60}\n")

    
    glob_start = time.time()
    all_mung_files = glob(gt_annotations_root + "/**/*.xml", recursive=True)
    mung_files_in_this_split = sorted([f for f in all_mung_files
                                        if os.path.splitext(os.path.basename(f))[0] in include_names])

    
    all_image_files = glob(images_root + "/**/*.png", recursive=True)
    image_files_in_this_split = sorted([f for f in all_image_files
                                         if os.path.splitext(os.path.basename(f))[0] in include_names])
    print(f"[DEBUG] File discovery took {time.time() - glob_start:.2f}s")

    print(f"Found {len(mung_files_in_this_split)} annotation files")
    print(f"Found {len(image_files_in_this_split)} image files\n")

    mungs = []
    images = []

    mung_load_times = []
    image_load_times = []

    for idx, (mung_file, image_file) in tqdm(enumerate(zip(mung_files_in_this_split, image_files_in_this_split)),
                                              total=len(mung_files_in_this_split)):
        mung_start = time.time()
        mung = __load_mung(mung_file, exclude_classes)
        mung_time = time.time() - mung_start
        mung_load_times.append(mung_time)

        num_nodes = len(mung.vertices)
        print(f"  [DEBUG] {idx+1}/{len(mung_files_in_this_split)}: {os.path.basename(mung_file)} - {num_nodes} nodes (loaded in {mung_time:.3f}s)")

        mungs.append(mung)

        image_start = time.time()
        image = __load_image(image_file)
        image_time = time.time() - image_start
        image_load_times.append(image_time)
        images.append(image)

        if max_items is not None and len(mungs) >= max_items:
            break

    total_time = time.time() - start_time
    print(f"\n✓ Loaded {len(mungs)} documents in {total_time:.2f}s")
    print(f"[DEBUG] Average mung load time: {np.mean(mung_load_times):.3f}s")
    print(f"[DEBUG] Average image load time: {np.mean(image_load_times):.3f}s\n")
    return mungs, images


def load_ground_truth_data(gt_annotations_root: str,
                           images_root: str,
                           split_file: str,
                           class_list: List[str],
                           class_dict: Dict[str, int],
                           config=None,
                           load_training_data=True,
                           load_validation_data=True,
                           load_test_data=False) -> Dict[str, GroundTruthDataPool]:
    """
    Load train/validation/test data pools using ground truth annotations.

    :param gt_annotations_root: Directory containing MUSCIMA++ XML files
    :param images_root: Directory containing image files
    :param split_file: YAML file defining train/val/test splits
    :param class_list: List of class names to include
    :param class_dict: Dictionary mapping class names to IDs
    :param config: Configuration dict with data pool parameters
    :param load_training_data: Whether to load training data
    :param load_validation_data: Whether to load validation data
    :param load_test_data: Whether to load test data
    :return: dict(train=training_pool, valid=validation_pool, test=test_pool)
    """
    split = load_split(split_file)

    data_pool_dict = config2data_pool_dict(config)

    validation_data_pool_dict = copy.deepcopy(data_pool_dict)
    if 'VALIDATION_MAX_NEGATIVE_EXAMPLES_PER_OBJECT' in config:
        validation_data_pool_dict['max_negative_samples'] = \
            config['VALIDATION_MAX_NEGATIVE_EXAMPLES_PER_OBJECT']

    training_pool = None
    validation_pool = None
    test_pool = None

    # Exclude classes not in class_list
    exclude_classes = [class_name for class_name in class_dict if class_name not in class_list]

    if load_training_data:
        print("Loading training data from ground truth...")
        tr_mungs, tr_images = __load_ground_truth_data(
            gt_annotations_root, images_root,
            include_names=split['train'],
            exclude_classes=exclude_classes
        )
        training_pool = GroundTruthDataPool(
            mungs=tr_mungs, images=tr_images,
            class_list=class_list, class_dict=class_dict,
            **data_pool_dict
        )

    if load_validation_data:
        print("Loading validation data from ground truth...")
        va_mungs, va_images = __load_ground_truth_data(
            gt_annotations_root, images_root,
            include_names=split['valid'],
            exclude_classes=exclude_classes
        )
     
        validation_data_pool_dict['filter_pairs'] = False
        validation_pool = GroundTruthDataPool(
            mungs=va_mungs, images=va_images,
            class_list=class_list, class_dict=class_dict,
            **validation_data_pool_dict
        )

    if load_test_data:
        print("Loading test data from ground truth...")
        te_mungs, te_images = __load_ground_truth_data(
            gt_annotations_root, images_root,
            include_names=split['test'],
            exclude_classes=exclude_classes
        )
  
        test_data_pool_dict = copy.deepcopy(data_pool_dict)
        test_data_pool_dict['filter_pairs'] = False
        test_pool = GroundTruthDataPool(
            mungs=te_mungs, images=te_images,
            class_list=class_list, class_dict=class_dict,
            **test_data_pool_dict
        )

    return dict(train=training_pool, valid=validation_pool, test=test_pool)
