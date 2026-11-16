from types import SimpleNamespace
import unittest
import torch
from check_ids import training_image_id,replace_image_id


class Tests(unittest.TestCase):
    def test_three_camera_ids_match_training(self):
        for cam in [0,1,2]:
            camera=SimpleNamespace(unique_img_idx=torch.arange(199)*3+cam)
            self.assertEqual(training_image_id(camera,24,597),72+cam)

    def test_mapping_is_authoritative_not_hardcoded(self):
        camera=SimpleNamespace(unique_img_idx=torch.tensor([7,11,19]))
        self.assertEqual(training_image_id(camera,1,30),11)

    def test_only_image_id_changes(self):
        clock=torch.tensor([.25]);rays=torch.randn(2,3,3);pose=torch.eye(4)
        data=dict(image_infos=dict(img_idx=torch.full((2,3),24,dtype=torch.long),normed_time=clock,origins=rays),cam_infos={'camera_to_world':pose})
        camera=SimpleNamespace(unique_img_idx=torch.arange(199)*3+2)
        self.assertEqual(replace_image_id(data,camera,24,597),74)
        self.assertTrue((data['image_infos']['img_idx']==74).all());self.assertEqual(data['image_infos']['img_idx'].shape,(2,3))
        self.assertIs(data['image_infos']['normed_time'],clock);self.assertIs(data['image_infos']['origins'],rays);self.assertIs(data['cam_infos']['camera_to_world'],pose)

    def test_out_of_bounds_embedding_rejected(self):
        with self.assertRaises(ValueError):training_image_id(SimpleNamespace(unique_img_idx=torch.tensor([999])),0,597)


if __name__=='__main__':unittest.main()
