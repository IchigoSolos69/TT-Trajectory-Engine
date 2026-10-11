import unittest
import numpy as np
from yolo_training_policy import (BalancedDomainSampler,camera_augment,checkpoint_fitness,center_square_labels)


class TrainingPolicyTests(unittest.TestCase):
    def records(self):
        return [dict(domain=d,positive=p,role="train") for d,p in
                [("match",True),("match",False),("iphone",True),("iphone",False),("pc_camera",True),("pc_camera",False)]]

    def test_exact_quota_and_live_class_balance(self):
        sampler=BalancedDomainSampler(self.records(),epoch_samples=96)
        counts=sampler.summary()
        self.assertEqual(sum(counts["match"].values()),48)
        for d in ("iphone","pc_camera"):
            self.assertEqual(counts[d],dict(positive=16,absent=8))

    def test_reproducible_but_epoch_changes(self):
        a,b=BalancedDomainSampler(self.records()),BalancedDomainSampler(self.records())
        self.assertEqual(list(a),list(b))
        old=list(a); a.set_epoch(1)
        self.assertNotEqual(old,list(a))

    def test_no_test_or_missing_group_sampling(self):
        records=self.records(); records[0]["role"]="test"
        with self.assertRaisesRegex(ValueError,"Only train"): BalancedDomainSampler(records)
        with self.assertRaisesRegex(ValueError,"both cameras"): BalancedDomainSampler(self.records()[:-1])

    def test_minimax_validation_ignores_domain_size(self):
        reports={d:dict(precision=.9,recall=.8,f1=v,frames=n) for d,v,n in
                 [("match",.99,100000),("iphone",.5,1000),("pc_camera",.8,1000)]}
        self.assertEqual(checkpoint_fitness(reports),.5)
        reports["iphone"].update(precision=None,recall=0.,f1=0.)
        self.assertEqual(checkpoint_fitness(reports),0.)
        reports["iphone"]["f1"]=None
        with self.assertRaisesRegex(ValueError,"Undefined"): checkpoint_fitness(reports)

    def test_extent_expands_with_centered_motion(self):
        image=np.zeros((64,64,3),dtype=np.uint8); image[30:34,30:34]=255
        boxes=np.array([[30,30,34,34]],dtype=np.float32)
        result,expanded,trace=camera_augment(image,boxes,seed=1,forced=dict(kernel_size=7,angle=0))
        np.testing.assert_allclose(expanded,[[27,30,37,34]])
        np.testing.assert_allclose((expanded[:,:2]+expanded[:,2:])/2,[[32,32]])
        self.assertEqual(result.shape,image.shape)
        self.assertFalse(trace["motion_skipped_boundary"])
        np.testing.assert_allclose(boxes,[[30,30,34,34]])

    def test_center_square_target_does_not_expand(self):
        image=np.full((64,64,3),127,dtype=np.uint8)
        _,boxes,_=camera_augment(image,[[20,20,30,30]],seed=5,extent_targets=False,forced=dict(kernel_size=5))
        np.testing.assert_allclose(boxes,[[20,20,30,30]])

    def test_motion_near_boundary_is_skipped(self):
        _,boxes,trace=camera_augment(np.zeros((64,64,3),dtype=np.uint8),[[0,20,4,24]],seed=4,forced=dict(kernel_size=7,angle=0))
        self.assertTrue(trace["motion_skipped_boundary"])
        np.testing.assert_allclose(boxes,[[0,20,4,24]])

    def test_policy_limits_and_absent_labels(self):
        image=np.zeros((16,16,3),dtype=np.uint8)
        _,boxes,_=camera_augment(image,[],seed=1)
        self.assertEqual(boxes.shape,(0,4))
        with self.assertRaisesRegex(ValueError,"outside"): camera_augment(image,[],seed=1,forced=dict(gamma=.1))
        with self.assertRaisesRegex(ValueError,"Invalid"): camera_augment(image,[[0,0,99,99]],seed=1)

    def test_target_convention_is_explicit_and_does_not_mutate(self):
        labels={"ball":[[.3,.3,.05,.05]]}
        squares=center_square_labels(labels,(1920,1080))
        self.assertAlmostEqual(squares["ball"][0][2]*1920,30)
        self.assertAlmostEqual(squares["ball"][0][3]*1080,30)
        self.assertEqual(labels["ball"][0][2],.05)


if __name__=="__main__": unittest.main()
