"""Regression checks for actual audit failures and segmentation/evaluation risks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import app
from vision.metrics import detection_score, match_detections
from vision.models import ModelLoadError, load_cnn
from vision.processing import (Component, box_mean, closing, correlate2d, dilate,
                               erode, grayscale)
from vision.real_data import read_real_manifest, real_image_path
from vision.real_evaluation import diagnose_matches, evaluate_real_cases
from vision.segmentation import region_proposals, select_regions
from vision.service import png_data


class AuditRegressions(unittest.TestCase):
    def test_invalid_morphology_cannot_change_image_dimensions(self):
        a = np.eye(7, 9, dtype=bool)
        for operation in (dilate, erode, closing):
            for bad in (0, -1, 2, 4, 1.5, True):
                with self.subTest(operation=operation.__name__, size=bad):
                    with self.assertRaises(ValueError): operation(a, bad)
            for size in (1, 3, 5): self.assertEqual(operation(a, size).shape, a.shape)
        with self.assertRaises(ValueError): correlate2d(a, np.ones((2, 3)))

    def test_integral_mean_matches_direct_windows(self):
        a = np.arange(35, dtype=np.float32).reshape(5, 7)
        padded = np.pad(a, 2, mode='edge')
        expected = np.array([[padded[y:y+5, x:x+5].mean() for x in range(7)] for y in range(5)])
        np.testing.assert_allclose(box_mean(a, 5), expected, atol=1e-6)
        np.testing.assert_allclose(box_mean(np.full((8, 9), 137), 101), 137)

    def test_diagnostics_use_same_one_to_one_matches_as_metrics(self):
        truth = [{'class':'circle','bbox':[10,10,20,20]} for _ in range(2)]
        objects = [{'class':'circle','bbox':[10,10,20,20],'target':True,'confidence':.99}]
        self.assertEqual(detection_score(truth, objects)[0]['circle'], {'tp':1,'fp':0,'fn':1})
        diagnostic = diagnose_matches(truth, objects)
        self.assertEqual(sum(d['failure_category']=='matched' for d in diagnostic), 1)
        self.assertEqual(diagnostic[1]['failure_category'], 'matching_region_already_assigned')
        self.assertEqual(len(match_detections(truth, objects)), 1)

    def test_corrupt_checkpoint_returns_actionable_load_error(self):
        with tempfile.TemporaryDirectory() as folder:
            weights=Path(folder)/'weights';weights.mkdir();(weights/'shapes_cnn.pt').write_bytes(b'broken weights')
            with self.assertRaises(ModelLoadError):load_cnn('shapes', Path(folder))

    def test_model_errors_are_json_503_for_all_prediction_routes(self):
        client=app.test_client();data={'image':png_data(np.zeros((20,20),np.uint8))}
        with patch('app.get_recognizer', side_effect=ModelLoadError('Khôi phục mô hình')):
            for route in ('/predict','/api/digits','/api/shapes'):
                response=client.post(route,json=data);self.assertEqual(response.status_code,503);self.assertIn('error',response.get_json())
            self.assertEqual(client.get('/api/health').status_code,503)

    def test_corrupt_evaluation_json_is_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir();(root/'results/evaluation.json').write_text('{invalid')
            with patch('app.ROOT',root):response=app.test_client().get('/api/metrics')
        self.assertEqual(response.status_code,503);self.assertIn('error',response.get_json())

    def test_stale_report_is_not_presented_as_current_metrics(self):
        from vision.provenance import asset_identity
        current=asset_identity()
        with patch('app.get_recognizer', return_value=SimpleNamespace(asset_identity=current)), patch('app.asset_identity',return_value={**current,'processing':'changed'}):
            response=app.test_client().get('/api/metrics')
        self.assertEqual(response.status_code,409);self.assertTrue(response.get_json()['stale'])

    def test_border_background_removed_before_cnn(self):
        image=np.full((80,100),240,np.uint8);image[20:60,30:70]=10
        candidates,trace=region_proposals(image,'global_regions')
        self.assertEqual(len(candidates),1);self.assertEqual(candidates[0][0].bbox,[30,20,40,40])
        self.assertEqual(trace['rejected']['frame_background'],1)

    def test_internal_detail_and_multibank_duplicate_are_removed(self):
        def candidate(x,y,w,h,bank):
            mask=np.ones((h,w),bool);return Component(x,y,w,h,w*h,mask),mask,bank
        candidates=[candidate(5,5,70,60,'global_dark'),candidate(20,20,15,15,'global_light'),candidate(5,5,70,60,'edges_otsu')]
        probabilities=np.array([[.001,.997,.001,.001]]*3)
        kept,rejected=select_regions(candidates,probabilities,['circle','rectangle','triangle','ellipse'],.75)
        self.assertEqual(len(kept),1);self.assertEqual(candidates[kept[0]][0].bbox,[5,5,70,60])
        self.assertEqual(rejected['internal_detail'],1);self.assertEqual(rejected['duplicate_region'],1)

    def test_development_evaluation_does_not_infer_new_test(self):
        seen=[]
        def checked_path(case,root):
            self.assertNotEqual(case['split'],'test');seen.append(case['id']);return real_image_path(case,root)
        fake=SimpleNamespace(profile={'mode':'global_regions'},geometry=lambda *args,**kwargs:{'objects':[],'warnings':[],'segmentation':{}})
        with patch('vision.real_evaluation.audit_real_manifest',return_value={}),patch('vision.real_evaluation.real_image_path',side_effect=checked_path),patch('vision.real_evaluation._roi_input',return_value=None):
            result=evaluate_real_cases(fake,ROOT,write=False)
        self.assertNotIn('test',result['detection_by_role']);self.assertEqual(len(seen),29)


if __name__=='__main__':unittest.main(verbosity=2)
