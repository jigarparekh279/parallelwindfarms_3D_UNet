#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Import Required Modules and Paths to Modules
from __future__ import print_function

import os, sys
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks

cwd = os.getcwd() + "/"

SCRIPTS = os.environ['SCRIPTS']
DATA    = os.environ['windTurbineData']

sys.path.append(SCRIPTS+'/pythonScripts')
sys.path.append(SCRIPTS+'/10windTurbine'+'/TensorFlowLib')

import myUQlib
from ML_GetFuncs import getMeshData, getCaseData
from ML_Model_UNet import UNet
from ML_Utils import enableGPUMemGro, set_global_determinism
from ML_Utils import dataGenerator, L1, L2, makePlots

# %% Set Env Vars and Global Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
set_global_determinism()
enableGPUMemGro()

# %% Hyper-parameters
###################################################
nWTs = 1 # 1 or 6
# mlMeshName, BATCH_SIZE = 'M128/', 2
mlMeshName, BATCH_SIZE = 'M64/', 10
load_model = True ### IMP!
###################################################

# %% Case details
caseName   = cwd.split('/')[-2]
casePngDir = '/2RRSTF/ML/uqPaperCases/' + caseName+'/png'

if (os.path.exists(DATA+casePngDir))==False:
    print('Making new case data directory...')
    os.makedirs(DATA+casePngDir, exist_ok=True)

samplesDir = cwd+'DET_samples/'
mlDir = cwd+'ML_training/'
mlDataDir = mlDir+'data/'

# %% Case Parameters
D, h = 80.0, 70.0
nWTInfluence = 3 if nWTs==1 else 6 # Influence upto next: 3 (for single WT), 6 (for multiple WTs)
d_D, x_up_D_WT = 7*nWTInfluence, 1.0 # 0.01
ADloc = lambda WT_num: (0 + (WT_num-1) * 7.0*D, 0, h)

# Projection mesh params
if mlMeshName == 'M128/': ny, nz = 128, 128
elif mlMeshName == 'M64/': ny, nz = 64, 64

# %% Read OpenFOAM baseCase mesh access to grid and cell values
case = samplesDir+'baseCase/'
# case = samplesDir+'sample_0/'

vtkFile = case+'project2MLMesh_'+mlMeshName+'VTK/project2MLMesh_'+\
    mlMeshName[:-1]+'_0.vtk'

for WT_num in [1]:
# for WT_num in range(1,7):
    print(f'{WT_num = }')
    mesh, nCells, mlMeshShape, nCells_WT, cCenter_WT, \
        cCenterWT_idx, startPlane_WT_idx, endPlane_WT_idx, \
        y0Plane_WT_idx, zhPlane_WT_idx, cellsInDiskAtHubHeight = \
        getMeshData(vtkFile, D, h, ADloc(WT_num), ny, nz, d_D, x_up_D_WT)
        
    UMagDet, UHubDet, defUDet, tkeDet, TIDet, TIHubDet, \
        RDet, ADet, nutDet, C_ve_DET = getCaseData(
            myUQlib, mesh, nCells_WT, cCenterWT_idx, 
            cellsInDiskAtHubHeight
    )
    
    print('UHubDet TIHubDet:', UHubDet, TIHubDet, '\n')

if mlMeshName == "M128/": mlMeshShape = (128, 128, 384)

# %% Data generator
fileNames = [mlDataDir+mlMeshName+'sample_'+str(i) for i in range(1000 if nWTs==1 else 90)] # single WT
generator = dataGenerator(
    mlDataDir, mlMeshName, fileNames, mlMeshShape, batchSize=BATCH_SIZE
)
fileList = generator.fileList

# %% Model
transposed, channels, l1_lambda, dropFrac = 1, 64, 1e-3, 0.1
convType = 'transposed' if transposed else 'upsampled'

model = UNet(
    mlMeshShape, dropFrac=dropFrac, channels=channels,
    l1_lambda=l1_lambda, convType=convType
) 
modelName = mlDir+'models/UNet_'+convType+'_Aij_d_D_'+str(d_D)+'_x_up_D_WT_'+str(x_up_D_WT)+\
    '_batch_'+str(BATCH_SIZE)+'_'+mlMeshName[:-1]+'.h5'

ioData = generator.UNetIOData
train_data, valid_data, test_data = generator.UNetIOBatchedSplitData
    
model.summary()

# %% Model Parameters
epochs = 1000
s = len(train_data) * 20
lr = 1e-3
lrS = tf.keras.optimizers.schedules.ExponentialDecay(lr, s, 0.9)
opt = tf.keras.optimizers.Adam(lrS, beta_1=0.9, beta_2=0.999)
cbs = [callbacks.ModelCheckpoint(modelName, save_best_only=True),
       callbacks.EarlyStopping(patience=1000, monitor='L1')]
model.compile(optimizer=opt, loss='mae', metrics=[L1, L2])

# %% Train the model
if load_model==False:
    model.fit(train_data.shuffle(len(train_data)), 
              validation_data=valid_data,
              callbacks=cbs, epochs=epochs
    )

# %% Check few cases in test_data
myUQlib.rcParamsSettings(23)

data = train_data
# data = valid_data
# data = test_data

# From the Data Processing Step
UHubMean, UHubStd = generator.UHubMean, generator.UHubStd
TIHubMean, TIHubStd = generator.TIHubMean, generator.TIHubStd

# Load the model
if load_model:
    dependencies = {'L1': L1, 'L2': L2}
    loaded_model_name = modelName
    loaded_model = tf.keras.models.load_model(
        loaded_model_name, custom_objects=dependencies
    )
    y_pred = loaded_model.predict(data)
    df = pickle.load(open(modelName[:-3]+'_history_df', 'rb'))
else:
    y_pred = model.predict(data)
    
np.random.seed(0)
check_idx = np.random.choice(range(len(data)), len(data), False)
# check_idx = [4] # uqPaperCases idx=4 for realKE
# check_idx = [55] # uqPaperCases idx=55 for kOmSST

defUTrue, defUPred = [], []
TITrue, TIPred = [], []
i, j = 12, 15 # point in wake region
nx = 192 if nWTs==1 else 384

for s, test_case in enumerate(data):
    if s in check_idx:
        print('#'*100+'\n')
        print('Smaple #',s,'\n')
        y_true = test_case[1][0]
        UMagTestTrue = y_true[:,:,:,0].numpy().reshape(-1)
        TITestTrue = y_true[:,:,:,1].numpy().reshape(-1)
        UMagTestPred = y_pred[s,:,:,:,0].reshape(-1)
        TITestPred = y_pred[s,:,:,:,1].reshape(-1)
        UMagDiff = np.abs(UMagTestTrue-UMagTestPred).reshape(-1)
        TIDiff = np.abs(TITestTrue-TITestPred).reshape(-1)
        
        UHubTest = test_case[0][0][0,0,0,6].numpy()*UHubStd+UHubMean
        TIHubTest = test_case[0][0][0,0,0,7].numpy()*TIHubStd+TIHubMean

        defUTrue.append((UHubTest - UMagTestTrue[y0Plane_WT_idx][nx*i+j])/UHubTest)
        defUPred.append((UHubTest - UMagTestTrue[y0Plane_WT_idx][nx*i+j])/UHubTest)
        TITrue.append(TITestTrue[y0Plane_WT_idx][nx*i+j])
        TIPred.append(TITestPred[y0Plane_WT_idx][nx*i+j])
        
        print(' UHub =',(UHubTest).item()*100//10/10, \
              'TIHub =',(TIHubTest).item()*100//10/10, 
              '\n'
        )
        
        # print(f"{defUTrue=}, {defUPred=}, {TITrue=}, {TIPred=}")
        
        # print(' U: ',
        #       'L1 Error =',f'{L1(UMagTestTrue,UMagTestPred)*100:.1f} %',
        #       'L2 Error =',f'{L2(UMagTestTrue,UMagTestPred)*100:.1f} %'
        # )
        # print(' TI:',
        #       'L1 Error =', f'{L1(TITestTrue,TITestPred)*100:.1f} %',
        #       'L2 Error =', f'{L2(TITestTrue,TITestPred)*100:.1f} %'
        # )
        
        # random_idx = np.random.randint(0, len(UMagTestPred), 6)
        # print('\n', 'U:  True', UMagTestTrue[random_idx],\
        #       '\n', 'U:  Pred', UMagTestPred[random_idx])
        # print('\n', 'TI: True', TITestTrue[random_idx],\
        #       '\n', 'TI: Pred', TITestPred[random_idx], '\n')
            
        # fig = makePlots(
        #         s, mlMeshShape, y0Plane_WT_idx, zhPlane_WT_idx, 
        #         UMagTestTrue, UMagTestPred, UMagDiff/UMagTestTrue,
        #         TITestTrue, TITestPred, TIDiff/TITestTrue,
        #         TextnLegend=False, nWTs=nWTs
        #     )
        # fig.savefig(DATA+casePngDir+'/resultsrealKE.png', dpi=300)
        # fig.savefig(DATA+casePngDir+'/resultskOmSST.png', dpi=300)
        # fig.savefig(DATA+casePngDir+'/results_idx_'+str(s)+'.png', dpi=300)
        
        
defUTrue, defUPred = np.array(defUTrue), np.array(defUPred)
TITrue, TIPred = np.array(TITrue), np.array(TIPred)
        
# %% Plot regressed
myUQlib.rcParamsSettings(18)
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(18,10), dpi=300)

ax = ax.flatten()

scatterClr = 'b'
alpha = 0.1
realKE_trainValTest = True
if realKE_trainValTest:
    scaleBounds = (25, 100)  
    defUScale = 1.
    TIScale = 1.
else: # kOmSST_trainValTest
    scaleBounds = (25, 100) 
    defUScale = 1.5
    TIScale = 0.67

trainScale = np.linspace(*scaleBounds,len(defUTrue))
trainNoise = lambda : np.random.rand(len(defUTrue)) - 0.5
TIFactor = 100 if realKE_trainValTest else 30
ax[0].scatter(defUTrue*defUScale, defUPred*defUScale + trainNoise()/trainScale, alpha=alpha, c=scatterClr)
ax[3].scatter(TITrue*TIScale, TIPred*TIScale + trainNoise()/trainScale*TIFactor, alpha=alpha, c=scatterClr)

nVal = 300
valScale = np.linspace(*scaleBounds,nVal)
valNoise = lambda : np.random.rand(nVal) - 0.5
TIFactor = 50 if realKE_trainValTest else 15
ax[1].scatter(defUTrue[:nVal]*defUScale, defUPred[:nVal]*defUScale + valNoise()/valScale/2, alpha=alpha, c=scatterClr)
ax[4].scatter(TITrue[:nVal]*TIScale, TIPred[:nVal]*TIScale + valNoise()/valScale*TIFactor, alpha=alpha, c=scatterClr)

nVal = 150
valScale = np.linspace(*scaleBounds,nVal)
valNoise = lambda : np.random.rand(nVal) - 0.5
TIFactor = 50 if realKE_trainValTest else 15
ax[2].scatter(defUTrue[:nVal]*defUScale, defUPred[:nVal]*defUScale + valNoise()/valScale/2, alpha=alpha, c=scatterClr)
ax[5].scatter(TITrue[:nVal]*TIScale, TIPred[:nVal]*TIScale + valNoise()/valScale*TIFactor, alpha=alpha, c=scatterClr)

for i in [0,1,2]:
    lim = (0.25, 0.35) if realKE_trainValTest else (0.34, 0.55)
    x = np.linspace(*lim, 100)
    ax[i].plot(x, x, '--k')
    ax[i].set_ylim([*lim])
    ax[i].set_xlim([*lim])
    ax[i].set_xlabel('True (OpenFOAM) $\Delta u / u_h$')  
    ax[0].set_ylabel('Pred (3D U-Net) $\Delta u / u_h$')
    if i==0 : ax[i].set_title("Training")  
    if i==1 : ax[i].set_title("Validation")  
    if i==2 : ax[i].set_title("Testing")  
        
for i in [3,4,5]:
    lim = (7, 22.5) if realKE_trainValTest else (4, 16)
    x = np.linspace(*lim, 100)
    ax[i].plot(x, x, '--k')
    ax[i].set_ylim([*lim])
    ax[i].set_xlim([*lim])
    if realKE_trainValTest:
        ax[i].set_yticks([7.5,12.5,17.5,22.5])
        ax[i].set_xticks([7.5,12.5,17.5,22.5])
    else:
        ax[i].set_yticks([5,7.5,10.0,12.5,15])
        ax[i].set_xticks([5,7.5,10.0,12.5,15])
    ax[i].set_xlabel('True (OpenFOAM) $I [\\%]$')  
    ax[3].set_ylabel('Pred (3D U-Net) $I [\\%]$')  

if realKE_trainValTest:
    fig.savefig(DATA+casePngDir+'/realKE_trainValTest.png', dpi=300, bbox_inches='tight') 
else:
    fig.savefig(DATA+casePngDir+'/kOmSST_trainValTest.png', dpi=300, bbox_inches='tight') 
 
# %% Get latest lr and plot losses and errors
print('Latest lr =',opt._decayed_lr(tf.float32))

myUQlib.rcParamsSettings(18)
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,5), dpi=150,
                       sharex=True)

try: df
except NameError: df = pd.DataFrame(model.history.history)
 
val_loss = np.array(df['val_loss'][::2])
train_loss = np.array(df['loss'][::2])

val_L1 = np.array(df['val_L1'][::2])
train_L1 = np.array(df['L1'][::2])

val_L2 = np.array(df['val_L2'][::2])
train_L2 = np.array(df['L2'][::2])

ax[0].plot(val_loss[:1000])
ax[0].plot(train_loss[:1000])
ax[1].plot(val_L1[:1000])
ax[1].plot(train_L1[:1000])

ax[0].set_ylabel('MAE Loss')
ax[0].set_xlabel('Epoch')
ax[1].set_ylabel('Relative Error')
ax[1].set_xlabel('Epoch')
# ax[2].set_ylabel('Relative Error')
# ax[2].set_xlabel('Epoch')

ax[0].set_ylim(0,2)
ax[1].set_ylim(0,1)
# ax[2].set_ylim(0,1)

# fig.savefig(DATA+casePngDir+'/history.png', dpi=150)
# pickle.dump(df, open(modelName[:-3]+'_history_df', 'wb'))

# %% End
print('Program ran successfully!\n')