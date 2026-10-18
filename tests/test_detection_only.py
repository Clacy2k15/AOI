"""偵測分支的判決／遮罩、失敗語意及模型安裝回歸檢查。"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deployment'))
from aoi.common import load_config, load_inference, read_json
from aoi.fusion import combine
from aoi.pipeline import Pipeline, save_records
from install_assets import extract_verified, import_models

def temporary():
    parent = ROOT / 'outputs/test-tmp'
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


THRESHOLDS = {'efficientad': {'image': 1., 'pixel': .5},
              'patchcore': {'image': 2., 'pixel': .7}}


def prediction(score, values):
    return {'score': score, 'map': np.asarray(values, dtype=np.float32)}


class FusionTests(unittest.TestCase):
    def test_image_thresholds_are_inclusive_or(self):
        for ea, pc, expected in [(0, 0, 'OK'), (1, 0, 'NG'), (0, 2, 'NG'), (1, 2, 'NG')]:
            with self.subTest(ea=ea, pc=pc):
                self.assertEqual(combine(prediction(ea, [[0]]), prediction(pc, [[0]]), THRESHOLDS)['decision'], expected)

    def test_mask_is_pixel_threshold_union(self):
        ea = prediction(0, [[.5, 0], [0, 0]])
        pc = prediction(0, [[0, .8], [0, 0]])
        result = combine(ea, pc, THRESHOLDS)
        np.testing.assert_array_equal(result['mask'], [[True, True], [False, False]])
        self.assertEqual(result['decision'], 'OK')

    def test_ng_can_have_empty_mask(self):
        result = combine(prediction(1, [[0]]), prediction(0, [[0]]), THRESHOLDS)
        self.assertEqual(result['decision'], 'NG')
        self.assertFalse(result['mask'].any())

    def test_mismatched_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            combine(prediction(0, [[0]]), prediction(0, [[0, 0]]), THRESHOLDS)


class ConfigurationTests(unittest.TestCase):
    def test_all_15_products_require_no_classifier(self):
        products = read_json(ROOT / 'outputs/expansion/product_registry.json')['products']
        self.assertEqual(len(products), 15)
        for product, row in products.items():
            with self.subTest(product=product):
                config = load_config(row['config'])
                settings = load_inference(config, require_calibrated=True)
                self.assertNotIn('classification', config)
                self.assertNotIn('qwen', settings)

    def test_runtime_assets_are_only_30_detector_checkpoints(self):
        files = read_json(ROOT / 'release-manifest.json')['files']
        self.assertEqual(len(files), 30)
        self.assertTrue(all(name.endswith('.ckpt') for name in files))
        self.assertFalse((ROOT / 'aoi/velm.py').exists())
        self.assertFalse((ROOT / 'scripts/start_qwen.py').exists())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary()
        self.directory = Path(self.temp.name)
        self.input = self.directory / 'input.png'
        self.rgb = np.zeros((5, 7, 3), dtype=np.uint8)
        cv2.imwrite(str(self.input), self.rgb)
        self.pipeline = Pipeline.__new__(Pipeline)
        self.pipeline.config = {}
        self.pipeline.product = 'bottle'
        self.pipeline.settings = {'thresholds': THRESHOLDS}
        class Stub:
            def __init__(self, score, values):
                self.result = prediction(score, values)
            def predict(self, rgb):
                return self.result
        ea_map = np.zeros((5, 7), dtype=np.float32)
        ea_map[1:3, 2:4] = .6
        self.pipeline.efficientad = Stub(1, ea_map)
        self.pipeline.patchcore = Stub(0, np.zeros((5, 7)))

    def tearDown(self):
        self.temp.cleanup()

    def test_outputs_binary_original_size_mask_without_classification(self):
        record = self.pipeline.predict(self.input, self.directory / 'results')
        self.assertEqual(record['decision'], 'NG')
        self.assertIsNone(record['error'])
        mask = cv2.imread(record['artifacts']['mask'], cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, self.rgb.shape[:2])
        self.assertEqual(set(np.unique(mask)), {0, 255})
        expected = self.pipeline.efficientad.result['map'] >= .5
        np.testing.assert_array_equal(mask, expected.astype(np.uint8) * 255)
        self.assertEqual(set(record), {'sample_id', 'input_path', 'product', 'decision', 'artifacts', 'error'})
        self.assertEqual(set(record['artifacts']), {'mask', 'overlay', 'json'})
        save_records([record], self.directory / 'batch')
        with (self.directory / 'batch/results.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]['decision'], 'NG')

    def test_failure_has_no_ok_ng_and_no_mask(self):
        with patch.object(self.pipeline.patchcore, 'predict', side_effect=RuntimeError('detector failed')):
            record = self.pipeline.predict(self.input, self.directory / 'results')
        self.assertIsNone(record['decision'])
        self.assertIn('detector failed', record['error'])
        self.assertNotIn('mask', record['artifacts'])

    def test_failed_image_write_does_not_report_success(self):
        with patch('aoi.pipeline.cv2.imwrite', return_value=False):
            record = self.pipeline.predict(self.input, self.directory / 'results')
        self.assertIsNone(record['decision'])
        self.assertIn('Could not save mask', record['error'])


def asset_record(data):
    return {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class InstallationTests(unittest.TestCase):
    def test_selective_extraction_skips_unneeded_archive_files(self):
        with temporary() as directory:
            root = Path(directory)
            contents = {'models/detector.ckpt': b'detector', 'unused.bin': b'unused'}
            archive = root / 'runtime.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                for name, data in contents.items():
                    output.writestr(name, data)
            expected = {name: asset_record(data) for name, data in contents.items()}
            selected = {'models/detector.ckpt': expected['models/detector.ckpt']}
            extract_verified(archive, root, expected, selected)
            self.assertEqual((root / 'models/detector.ckpt').read_bytes(), b'detector')
            self.assertFalse((root / 'unused.bin').exists())

    def test_corrupt_archive_preserves_existing_model(self):
        with temporary() as directory:
            root = Path(directory)
            (root / 'model.ckpt').write_bytes(b'existing')
            with zipfile.ZipFile(root / 'runtime.zip', 'w') as output:
                output.writestr('model.ckpt', b'wrong')
            with self.assertRaises(ValueError):
                extract_verified(root / 'runtime.zip', root, {'model.ckpt': asset_record(b'valid')})
            self.assertEqual((root / 'model.ckpt').read_bytes(), b'existing')
            self.assertFalse((root / 'model.ckpt.installing').exists())

    def test_path_traversal_rejected(self):
        with temporary() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / 'runtime.zip', 'w') as output:
                output.writestr('../outside.ckpt', b'bad')
            with self.assertRaises(ValueError):
                extract_verified(root / 'runtime.zip', root, {'../outside.ckpt': asset_record(b'bad')})

    def test_import_does_not_modify_source(self):
        with temporary() as directory:
            root = Path(directory) / 'target'
            source = Path(directory) / 'source'
            root.mkdir()
            source.mkdir()
            original = source / 'model.ckpt'
            original.write_bytes(b'weights')
            before = original.stat().st_mtime_ns
            (root / 'release-manifest.json').write_text(json.dumps({'files': {'model.ckpt': asset_record(b'weights')}}))
            import_models(root, source)
            self.assertEqual((root / 'model.ckpt').read_bytes(), b'weights')
            self.assertEqual(original.stat().st_mtime_ns, before)
            self.assertEqual(original.read_bytes(), b'weights')


if __name__ == '__main__':
    unittest.main()
