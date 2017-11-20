import pandas as pd
import numpy as np
from svmutil import *
import glob
import os

def check_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def check_int(x):
   
    check = x.isdigit()

    return check

files = glob.glob('data_files/*dt')

results = {}
count = 0

for name in files:


    sname = name.strip('data_files.')
    sname = sname[1:]
    
    print('Running Classification task on the file ',sname)

    pdata = pd.read_csv(name,header=None,nrows=7,delimiter=' ')
    dp = pdata.as_matrix()
    din = int(dp[0,0].strip('bool_in='))  #length of inputs
    rout = int(dp[3,0].strip('real_out='))
    
    if din != 0 or rout !=0:
        
        print(sname,'.dt format is not supported')
        results[count] = [sname,'File format not supported']
        count += 1
        
    else:
            
        lin = int(dp[1,0].strip('real_in='))  #length of inputs
        lout = int(dp[2,0].strip('bool_out='))        
        tr = int(dp[4,0].strip('training_examples='))
        val = int(dp[5,0].strip('validation_examples='))
        te = int(dp[6,0].strip('testing_examples='))

        print('real in=',lin,'\nbool out =',lout,'\n',tr,'\n',val,'\n',te)

        #print(stop)
        
        data = pd.read_csv(name,header=None,skiprows=7,delimiter=' ')
        d = data.as_matrix()
        row,col = d.shape
        
        for r in range(row):
            for co in range(col):
                if not (check_float(d[r,co]) or check_int(d[r,co])):
                    d[r,co] = '0'
            
        gotc = np.array(d).astype(np.float32)

        lbls = []

        for k in range(len(gotc)):
            atmp = gotc[k]
            label = np.argmax(atmp[lin:])
            lbls.append(label)

        #print(lbls)
        #print(stop)


        flen = gotc.shape[1]
        lim = flen-1
        #lbls = gotc[:,lim].flatten().tolist()
        feats = np.delete(gotc,[lim],axis=1).tolist()

        trfeat = feats[:tr]
        trlbl = lbls[:tr]

        valfeat = feats[tr:(tr+val)]
        valbl = lbls[tr:(tr+val)]

        tefeat = feats[(tr+val):]
        telbl = lbls[(tr+val):]

        print(len(trlbl),len(valbl),len(telbl))


        print('Training SVM')

        #grid search and cross validation for good parameters of RBF
        gridC = [2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),2**(0),2**(1),2**(2),2**(3),2**(4),2**(5),2**(6),2**(7),2**(8),2**(9),2**(10),2**(11),2**(12),2**(13),2**(14),2**(15)]
        gridg = [2**(4),2**(5),2**(-15),2**(-14),2**(-13),2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),2**(0),2**(1),2**(2),2**(3),2**(4),2**(5)]
        highest = 0         #highest accuracy attained
        fparam = '' #final set of parameters
        folder = os.path.dirname('models/')
        if not os.path.exists(folder):
            os.makedirs(folder)
        for C in range(len(gridC)):
            for g in range(len(gridg)):

                    gp = gridg[g]
                    Cp = gridC[C]

                    param = '-t 2'+' -g '+ str(gp)+' -c '+str(Cp)+' -b 1'            #form parameter string
                    m = svm_train(trlbl, trfeat,param)       #cross validation training
                    p_label, p_acc, p_val = svm_predict(valbl, valfeat, m)

                    if(p_acc[0] > highest):  #get and store parameters producing highest cross validation accuracy
                        highest = p_acc[0]
                        fparam = param
                        svm_save_model('models/'+sname+'.model',m)

                    #print('\n')    
                    if(highest == 100): break;  #break if parameters found give 100% cross validation accuracy
            if(highest == 100): break;

        print('Best Validation accuracy {0}%'.format(highest))

        print('Evaluating Test data of ',sname,'dt')

        m = svm_load_model('models/'+sname+'.model')

        p_label, p_acc, p_val = svm_predict(telbl,tefeat, m,'-b 1')

        results[count] = [sname,p_acc[0]]

        count += 1
        
print('==========================================================')
print('\n\nTest Evaluation Results...\n\n')

for i in range(len(results)):

    re = results[i]
    print(re[0],'.dt accuracy = ',re[1],'\n\n')



