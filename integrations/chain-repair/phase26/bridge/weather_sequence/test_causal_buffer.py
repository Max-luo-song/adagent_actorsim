import unittest
import numpy as np
from causal_buffer import CausalWeatherBuffer

class CausalTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((3,8,12,3),30,dtype=np.uint8)
        self.mask=np.zeros((3,8,12),dtype=np.uint8)
    def predict(self,x,m):return np.full_like(x,int(x.mean()))
    def run_stream(self,values):
        b=CausalWeatherBuffer();out=[]
        for i,v in enumerate(values):
            out.append(b.process(i,i*.1,'epoch0',self.rgb+v,self.mask,self.predict))
        return out
    def test_future_perturbation_cannot_change_committed_prefix(self):
        a=self.run_stream([0,10,20]);b=self.run_stream([0,10,150])
        for i in (0,1):np.testing.assert_array_equal(a[i][0],b[i][0])
        self.assertFalse(np.array_equal(a[2][0],b[2][0]))
    def test_history_padding_and_last_triplet(self):
        b=CausalWeatherBuffer();_,audit=b.process(80,8.,'a',self.rgb,self.mask,lambda x,m:x)
        self.assertEqual(audit['window_frames'],[80,80,80])
        def labeled(x,m):
            y=x.copy()
            for i in range(9):y[i]=i
            return y
        out,audit=b.process(81,8.1,'a',self.rgb,self.mask,labeled)
        self.assertEqual(audit['window_frames'],[80,80,81])
        self.assertEqual(out[:,0,0,0].tolist(),[6,7,8])
    def test_epoch_change_discards_old_history(self):
        b=CausalWeatherBuffer();b.process(80,8.,'a',self.rgb,self.mask,self.predict)
        _,audit=b.process(81,8.1,'edited',self.rgb,self.mask,self.predict)
        self.assertEqual(audit['window_frames'],[81,81,81])
    def test_failed_inference_does_not_advance_history(self):
        b=CausalWeatherBuffer()
        def fail(x,m):raise RuntimeError('inference failed')
        with self.assertRaises(RuntimeError):b.process(80,8.,'a',self.rgb,self.mask,fail)
        _,audit=b.process(80,8.,'a',self.rgb,self.mask,self.predict)
        self.assertEqual(audit['window_frames'],[80,80,80])
    def test_input_mutation_does_not_modify_history(self):
        b=CausalWeatherBuffer();b.process(80,8.,'a',self.rgb,self.mask,self.predict)
        self.rgb[:]=90
        def inspect(x,m):
            self.assertTrue((x[:6]==30).all());return x
        b.process(81,8.1,'a',self.rgb,self.mask,inspect)
    def test_non_monotone_time_rejected(self):
        b=CausalWeatherBuffer();b.process(80,8.,'a',self.rgb,self.mask,self.predict)
        with self.assertRaises(ValueError):b.process(81,7.9,'a',self.rgb,self.mask,self.predict)

if __name__=='__main__':unittest.main()
