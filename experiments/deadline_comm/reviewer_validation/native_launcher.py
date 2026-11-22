"""Author SchedNet source with runtime API compatibility, no algorithm rewrite.

TensorFlow 2.15 compat.v1, Python2 ConfigParser alias, seed inside new TF graphs.
Run in a unique output cwd; author files remain byte-identical to source manifest.
"""
import argparse
import configparser
import json
import os
from pathlib import Path
import random
import sys
import time

ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--control',choices=['native','round_robin'],default='native');ap.add_argument('--testing-steps',type=int,default=100)
a=ap.parse_args()
source=Path(__file__).resolve().parent/'native_schednet';out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'results/eval').mkdir(parents=True);(out/'results/nn').mkdir(parents=True)
os.chdir(out);sys.path.insert(0,str(source));os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow']=tf;sys.modules['ConfigParser']=configparser
OldGraph=tf.Graph
class SeededGraph(OldGraph):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.seed=a.seed
tf.Graph=SeededGraph
sys.argv=['native_schednet','--seed='+str(a.seed),'--training_step='+str(a.steps),'--testing_step='+str(a.testing_steps),'--eval_on_train=False','--gui=False','--max_step=50']
random.seed(a.seed);np.random.seed(a.seed);tf.set_random_seed(a.seed)
start=time.monotonic()
manifest=dict(status='starting',source_manifest=json.loads((source/'SOURCE_MANIFEST.json').read_text()),seed=a.seed,steps=a.steps,control=a.control,
              compatibility=['tf.compat.v1','ConfigParser alias','per-graph random seed'],
              scope='Native author Predator-Prey training diagnostic, not matched Hallway/deadline comparison',device='CPU')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
try:
    import config
    import make_env
    from agents.schednet.trainer import Trainer
    trainer=Trainer(make_env.make_env(config.flags.FLAGS.scenario))
    if a.control=='round_robin':
        import types
        def fixed_schedule(self,obs_n,global_step,type,train=True):
            k=config.flags.FLAGS.s_num
            selected=[(global_step+i)%self._n_predator for i in range(k)]
            mask=np.zeros(self._n_predator);mask[selected]=1.
            return mask,mask.copy()
        trainer.get_schedule=types.MethodType(fixed_schedule,trainer)
        manifest['control_change']='Scheduling replaced by deterministic rotating slots; native networks and learner retained, including unused weight generator updates'
    trainer.learn();trainer.test()
    manifest.update(status='completed',seconds=time.monotonic()-start)
except Exception as exc:
    manifest.update(status='failed',error=repr(exc));raise
finally:(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
