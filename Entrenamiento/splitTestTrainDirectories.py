import os
#path varia de ubicacion de data
source1 = "spoken_numbers_wav/mfcc_image_ts/MauricioPesantez"
dest11 = "spoken_numbers_wav/mfcc_image_tr/MauricioPesantez"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
    if np.random.rand(1) < 0.2:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)
