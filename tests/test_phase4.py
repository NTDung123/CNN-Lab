"""Phase-4 regressions: displayed masks, projected labels, data roles and CNN."""
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vision.processing import Component
from vision.segmentation import convex_hull_mask,region_proposals,select_regions
from vision.datasets import shape_fingerprint
from vision.metrics import classification_metrics
from vision.real_data import audit_real_manifest,read_real_manifest
from scripts.evaluate import _scene_input_hashes


class Phase4Tests(unittest.TestCase):
    def test_preview_does_not_union_foreground_and_complement(self):
        image=np.full((100,120),240,np.uint8);image[30:70,40:80]=20
        candidates,trace=region_proposals(image)
        self.assertEqual(len(candidates),1)
        self.assertGreater(trace['joined'].mean(),0);self.assertLess(trace['joined'].mean(),.25)
        self.assertEqual(len(trace['branches']),2)
        np.testing.assert_array_equal(trace['branches'][0]['binary']|trace['branches'][1]['binary'],True)
        np.testing.assert_array_equal(trace['joined'],trace['branches'][0]['candidates'])

    def test_circle_tangent_to_four_sides_is_not_frame_background(self):
        y,x=np.indices((65,65));circle=(x-32)**2+(y-32)**2<=32.2**2
        image=np.where(circle,20,240).astype(np.uint8)
        candidates,_=region_proposals(image,'global_regions')
        self.assertTrue(any(c.bbox==[0,0,65,65] and np.count_nonzero(mask^circle)<30 for c,mask,_ in candidates))

    def test_convex_hull_repairs_a_notch_without_expanding_bbox(self):
        mask=np.zeros((30,40),bool);mask[5:25,8:32]=True;mask[5:15,16:24]=False
        original=mask.copy();hull=convex_hull_mask(mask)
        self.assertTrue(hull[5:25,8:32].all());self.assertEqual(int(hull.sum()),20*24)
        np.testing.assert_array_equal(mask,original)

    def test_background_cnn_cannot_suppress_real_object_as_parent(self):
        big=np.ones((90,90),bool);small=np.ones((20,30),bool)
        candidates=[(Component(0,0,90,90,8100,big),big,'local_dark'),(Component(20,20,30,20,600,small),small,'global_dark')]
        probabilities=np.array([[.0001,.0001,.0001,.0001,.9996],[.0001,.9996,.0001,.0001,.0001]])
        selected,rejected=select_regions(candidates,probabilities,['circle','rectangle','triangle','ellipse','background'],.75)
        self.assertEqual(selected,[1]);self.assertEqual(rejected['cnn_background'],1)

    def test_real_validation_has_both_classes_and_negative_frames(self):
        audit=audit_real_manifest(ROOT)
        self.assertGreater(audit['class_counts_by_role']['validation']['circle'],0)
        self.assertGreater(audit['class_counts_by_role']['validation']['rectangle'],0)
        self.assertEqual(audit['negative_frames_by_role']['validation'],2)
        self.assertEqual(audit['negative_frames_by_role']['test'],2)

    def test_old_test_is_development_and_projected_ellipse_is_not_circle(self):
        manifest=read_real_manifest(ROOT);cases={c['id']:c for c in manifest['cases']}
        for c in manifest['cases']:
            if c.get('previous_phase3_role')=='test':self.assertEqual(c['split'],'development')
        for name in ('cd_gray','gold_cap'):
            self.assertEqual(cases[name]['annotations'],[]);self.assertEqual(cases[name]['distractors'][0]['class'],'ellipse')

    def test_hybrid_training_is_unique_and_cannot_use_real_test(self):
        meta=json.loads((ROOT/'data/real_training.json').read_text())
        self.assertEqual(meta['real_test_cases_opened'],[])
        self.assertTrue(all(c['role'] in ('development','validation') for c in meta['source_cases_opened']))
        with np.load(ROOT/'data/real_training.npz',allow_pickle=False) as d:
            train=[shape_fingerprint(x) for x in d['x_train']];val={shape_fingerprint(x) for x in d['x_real_validation']}
            self.assertEqual(len(train),len(set(train)));self.assertFalse(set(train)&val)
            self.assertFalse(set(d['groups_train'])&set(d['groups_real_validation']))
            self.assertGreater(np.count_nonzero(d['y_train']==4),1000)

    def test_macro_f1_not_reduced_by_absent_background_output(self):
        metrics=classification_metrics([0,1,2,3],[0,1,2,3],['circle','rectangle','triangle','ellipse','background'])
        self.assertEqual(metrics['macro_f1'],1.);self.assertEqual(len(metrics['macro_average_classes']),4)

    def test_scene_audit_hashes_every_actual_proposal_before_inference(self):
        images=[np.eye(32,dtype=np.float32),np.ones((32,32),np.float32)]
        fake=SimpleNamespace(prepare_geometry=lambda image:{'normalized':images})
        self.assertEqual(_scene_input_hashes(None,fake),[shape_fingerprint(x) for x in images])


if __name__=='__main__':unittest.main(verbosity=2)
