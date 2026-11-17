from pathlib import Path
import sys
import numpy as np
out=Path(sys.argv[2])
orig=np.zeros(2058,np.int32);orig[8]=40;orig[10:]=np.arange(2048)
np.save(out/'orig.npy',orig)
for pos in (0, 1, 14, 15, 16, 31, 32, 63, 64, 1024, 31999, 32000, 32766):
 info=np.array([[0,pos,1,0]],np.int32);np.save(out/f'info_{pos}.npy',info)
 visible=pos+1;s=min((visible+31)//32,64);per=(visible+s-1)//s
 expected=np.zeros((1,14,64,32,2),np.float32)
 for split in range(s):
  count=max(0,min((split+1)*per,visible)-min(split*per,visible))
  value=np.float32(0)
  for _ in range(count):value=np.float32(value+np.float32(np.float16(0.01)))
  expected[:,:,split,:,:]=value
 np.save(out/f'expected_{pos}.npy',expected.reshape(-1))
