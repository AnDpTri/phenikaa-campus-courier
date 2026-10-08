from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from courier.common import load_dataset
from courier.cv import (
    CVInput,
    CVPipeline,
    CVReport,
    LegendReading,
    OracleEdgeClassifier,
    OracleGridDetector,
    OracleLegendReader,
    OracleNodeClassifier,
    OracleWeatherClassifier,
    crop_edge_strip,
    crop_square,
    load_annotations,
    scene_image_paths,
    validate_graph,
)
from courier.cv.features import extract_weather_features
from courier.cv.learned import SklearnWeatherClassifier
from courier.cv.detector import DetectorNet, KeypointDetector, letterbox
from courier.cv.nets import load_net, save_net


DATA_ROOT = Path(__file__).parents[2] / "Phenikaa_Campus_Courier_2026_v3" / "delivery_public"


class AnnotationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.annotations = load_annotations(DATA_ROOT, "validation")
        cls.dataset = load_dataset(DATA_ROOT, "validation")

    def test_oracle_pipeline_rebuilds_common_scene(self) -> None:
        assert self.dataset.scenes is not None
        pipeline = CVPipeline(
            legend=OracleLegendReader(self.annotations),
            weather=OracleWeatherClassifier(self.annotations),
            grid=OracleGridDetector(self.annotations),
            edges=OracleEdgeClassifier(self.annotations),
            nodes=OracleNodeClassifier(self.annotations),
        )
        report = CVReport()
        blank = np.zeros((1, 1, 3), dtype=np.uint8)
        for annotation, scene in zip(self.annotations, self.dataset.scenes, strict=True):
            graph = pipeline.extract(CVInput(scene_id=annotation.scene_id, image=blank))
            self.assertEqual(validate_graph(graph), [])
            self.assertEqual(graph.to_scene(scene.scene_id, scene.mission), scene)
            report.add(graph, annotation.graph)
        summary = report.summary()
        self.assertEqual(summary["scene_exact"], 1.0)
        self.assertEqual(summary["node_err_px_mean"], 0.0)

    def test_legend_road_rows_have_fixed_order(self) -> None:
        for annotation in self.annotations:
            kinds = [entry.kind for entry in annotation.legend.entries if entry.kind in
                     ("normal", "crowded", "covered", "closed", "stairs", "oneway", "robot")]
            self.assertEqual(kinds, ["normal", "crowded", "covered", "closed", "stairs", "oneway", "robot"])
            self.assertEqual(annotation.legend.place_types, {kind for kind, _ in annotation.graph.landmarks})

    def test_image_paths_match_observations(self) -> None:
        paths = scene_image_paths(DATA_ROOT, "validation")
        self.assertEqual([scene_id for scene_id, _ in paths], [a.scene_id for a in self.annotations])
        self.assertEqual([path for _, path in paths], [a.image_path for a in self.annotations])


class CropGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        # Red square at the left end, blue square at the right end of a horizontal segment.
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.image[45:55, 40:50] = (255, 0, 0)
        self.image[45:55, 150:160] = (0, 0, 255)

    def test_edge_strip_maps_a_left_b_right_in_every_direction(self) -> None:
        a, b = (45.0, 50.0), (155.0, 50.0)
        forward = crop_edge_strip(self.image, a, b, thickness=20, out_size=(110, 20))
        backward = crop_edge_strip(self.image, b, a, thickness=20, out_size=(110, 20))
        self.assertGreater(forward[10, 3, 0], 200)
        self.assertGreater(forward[10, -4, 2], 200)
        self.assertGreater(backward[10, 3, 2], 200)
        self.assertGreater(backward[10, -4, 0], 200)

    def test_vertical_strip_is_rotated_to_horizontal(self) -> None:
        rotated = np.ascontiguousarray(np.rot90(self.image, k=-1))  # segment now runs top -> bottom
        h = self.image.shape[0]
        a, b = (h - 1 - 50.0, 45.0), (h - 1 - 50.0, 155.0)
        strip = crop_edge_strip(rotated, a, b, thickness=20, out_size=(110, 20))
        self.assertEqual(strip.shape, (20, 110, 3))
        self.assertGreater(strip[10, 3, 0], 200)
        self.assertGreater(strip[10, -4, 2], 200)

    def test_square_crop_is_centred_and_padded(self) -> None:
        patch = crop_square(self.image, (45.0, 50.0), side=10, out_size=10)
        self.assertTrue((patch[2:8, 2:8] == (255, 0, 0)).all())
        corner = crop_square(self.image, (0.0, 0.0), side=40, out_size=40)
        self.assertEqual(corner.shape, (40, 40, 3))


class DetectorArtifactTests(unittest.TestCase):
    def test_letterbox_supports_higher_resolution(self) -> None:
        image = np.zeros((480, 960, 3), dtype=np.uint8)
        boxed, scale = letterbox(image, 768)
        self.assertEqual(boxed.shape, (768, 768, 3))
        self.assertEqual(scale, 0.8)

    def test_detector_input_size_round_trips_in_artifact_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "detector.pt"
            save_net(DetectorNet(8), path, init={"width": 8}, input_size=768)
            loaded = load_net(path)
            self.assertEqual(loaded.artifact_meta["input_size"], 768)
            detector = KeypointDetector(loaded, input_size=loaded.artifact_meta["input_size"])
            heat, _, _ = detector.predict_maps(np.zeros((80, 120, 3), dtype=np.uint8))
            self.assertEqual(heat.shape[-2:], (192, 192))


class WeatherClassifierTests(unittest.TestCase):
    class ConstantModel:
        def predict(self, features):
            assert features.shape[0] == 1
            assert np.isfinite(features).all()
            return np.asarray(["rain"])

    @staticmethod
    def artifact(model) -> dict:
        return {
            "kind": SklearnWeatherClassifier.ARTIFACT_KIND,
            "artifact_version": SklearnWeatherClassifier.ARTIFACT_VERSION,
            "feature_version": 1,
            "crop_pad": 4.0,
            "model": model,
        }

    def test_features_have_fixed_shape_across_crop_sizes(self) -> None:
        small = extract_weather_features(np.zeros((20, 31, 3), dtype=np.uint8))
        large = extract_weather_features(np.zeros((101, 87, 3), dtype=np.uint8))
        self.assertEqual(small.shape, large.shape)
        self.assertGreater(small.size, 1_000)

    def test_artifact_classifier_uses_weather_box(self) -> None:
        classifier = SklearnWeatherClassifier(self.artifact(self.ConstantModel()))
        cv_input = CVInput(scene_id="synthetic", image=np.zeros((80, 80, 3), dtype=np.uint8))
        legend = LegendReading(entries=(), weather_box=(20.0, 20.0, 60.0, 60.0))
        self.assertEqual(classifier.classify(cv_input, legend), "rain")

    def test_missing_weather_box_is_explicit(self) -> None:
        classifier = SklearnWeatherClassifier(self.artifact(self.ConstantModel()))
        cv_input = CVInput(scene_id="synthetic", image=np.zeros((20, 20, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "weather icon was not located"):
            classifier.classify(cv_input, LegendReading(entries=(), weather_box=None))


if __name__ == "__main__":
    unittest.main()
