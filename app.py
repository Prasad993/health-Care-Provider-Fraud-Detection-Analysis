from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Dtree.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('deployment.html')

@app.route("/predict", methods=['POST'])
def predict():
    c=['Provider', 'BeneID', 'Gender', 'Race',
        'RenalDiseaseIndicator', 'State', 'County', 'ChronicCond_Alzheimer',
        'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
        'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
        'ChronicCond_Depression', 'ChronicCond_Diabetes',
        'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
        'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
        'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
        'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'InscClaimAmtReimbursed',
        'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician',
        'ClmAdmitDiagnosisCode', 'DeductibleAmtPaid',
        'DiagnosisGroupCode', 'ClmDiagnosisCode_1',
        'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
        'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
        'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
        'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
        'ClmProcedureCode_4',"ClaimDuration"]  
    if request.method == 'POST':
        Provider=request.form['Provider']
        BeneID=request.form['BeneID']
        Gender=int(request.form['Gender'])
        Race=int(request.form['Race'])
        RenalDiseaseIndicator=request.form['RenalDiseaseIndicator']
        State=int(request.form['State'])
        County=int(request.form['County'])
        ChronicCond_Alzheimer=request.form['ChronicCond_Alzheimer']
        ChronicCond_Heartfailure=request.form['ChronicCond_Heartfailure']
        ChronicCond_KidneyDisease=request.form['ChronicCond_KidneyDisease']
        ChronicCond_Cancer=request.form['ChronicCond_Cancer']
        ChronicCond_ObstrPulmonary=request.form['ChronicCond_ObstrPulmonary']
        ChronicCond_Depression=request.form['ChronicCond_Depression']
        ChronicCond_Diabetes=request.form['ChronicCond_Diabetes']
        ChronicCond_IschemicHeart=request.form['ChronicCond_IschemicHeart']
        ChronicCond_Osteoporasis=request.form['ChronicCond_Osteoporasis']
        ChronicCond_rheumatoidarthritis=request.form['ChronicCond_rheumatoidarthritis']
        ChronicCond_stroke=request.form['ChronicCond_stroke']
        IPAnnualReimbursementAmt=int(request.form['IPAnnualReimbursementAmt'])
        IPAnnualDeductibleAmt=int(request.form['IPAnnualDeductibleAmt'])
        OPAnnualReimbursementAmt=int(request.form['OPAnnualReimbursementAmt'])
        OPAnnualDeductibleAmt=int(request.form['OPAnnualDeductibleAmt'])
        InscClaimAmtReimbursed=int(request.form['InscClaimAmtReimbursed'])
        AttendingPhysician=request.form['AttendingPhysician']
        OperatingPhysician=request.form['OperatingPhysician']
        OtherPhysician=request.form['OtherPhysician']
        DeductibleAmtPaid=request.form['DeductibleAmtPaid']
        DiagnosisGroupCode=request.form['DiagnosisGroupCode']
        ClmDiagnosisCode_1=request.form['ClmDiagnosisCode_1']
        ClmDiagnosisCode_2=request.form['ClmDiagnosisCode_2']
        ClmDiagnosisCode_3=request.form['ClmDiagnosisCode_3']
        ClmDiagnosisCode_4=request.form['ClmDiagnosisCode_4']
        ClmDiagnosisCode_5=request.form['ClmDiagnosisCode_5']
        ClmDiagnosisCode_6=request.form['ClmDiagnosisCode_6']
        ClmDiagnosisCode_7=request.form['ClmDiagnosisCode_7']
        ClmDiagnosisCode_8=request.form['ClmDiagnosisCode_8']
        ClmDiagnosisCode_9=request.form['ClmDiagnosisCode_9']
        ClmDiagnosisCode_10=request.form['ClmDiagnosisCode_10']
        ClmProcedureCode_1=request.form['ClmProcedureCode_1']
        ClmProcedureCode_2=request.form['ClmProcedureCode_2']
        ClmProcedureCode_3=request.form['ClmProcedureCode_3']
        ClmProcedureCode_4=request.form['ClmProcedureCode_4']
        ClmAdmitDiagnosisCode=request.form['ClmAdmitDiagnosisCode']

        Claim_duration=request.form['Claim_duration']        
        
        import pandas as pd
        data=[Provider, BeneID, Gender, Race,
        RenalDiseaseIndicator, State, County, ChronicCond_Alzheimer,
        ChronicCond_Heartfailure, ChronicCond_KidneyDisease,
        ChronicCond_Cancer, ChronicCond_ObstrPulmonary,
        ChronicCond_Depression, ChronicCond_Diabetes,
        ChronicCond_IschemicHeart, ChronicCond_Osteoporasis,
        ChronicCond_rheumatoidarthritis, ChronicCond_stroke,
        IPAnnualReimbursementAmt, IPAnnualDeductibleAmt,
        OPAnnualReimbursementAmt, OPAnnualDeductibleAmt, InscClaimAmtReimbursed,
        AttendingPhysician, OperatingPhysician, OtherPhysician,
        ClmAdmitDiagnosisCode, DeductibleAmtPaid,
        DiagnosisGroupCode, ClmDiagnosisCode_1,
        ClmDiagnosisCode_2, ClmDiagnosisCode_3, ClmDiagnosisCode_4,
        ClmDiagnosisCode_5, ClmDiagnosisCode_6, ClmDiagnosisCode_7,
        ClmDiagnosisCode_8, ClmDiagnosisCode_9, ClmDiagnosisCode_10,
        ClmProcedureCode_1, ClmProcedureCode_2, ClmProcedureCode_3,
        ClmProcedureCode_4, Claim_duration ]
        import numpy as np
        data=np.asarray(data)
        df=pd.DataFrame([data], columns=c )

        import pickle
        with open("patient_count.json","rb") as file:
            patient_count=pickle.load(file)
        with open("provider_count.json","rb") as file:
            provider_count=pickle.load(file)
        with open("attending_phy_count.json","rb") as file:
            att_phy_count=pickle.load(file)

        sccale_cols=["Physician_count","DeductibleAmtPaid",'InscClaimAmtReimbursed',"amount_to_reimbursed",
                 'Race','State','County','AttendingPhysician','BeneID','Provider', 'amount_to_reimbursed']
        d_codes=[4019, 25000, 2724, 'V5869', 4011]
        p_codes=[4019.0, 9904.0, 2724.0, 8154.0, 66.0]



        def data_for_test(x):
            x["RenalDiseaseIndicator"]=np.where(x["RenalDiseaseIndicator"]=="Y",1,0) 
            x["BeneID"]=x["BeneID"].map(patient_count)
            x["Provider"]=x["Provider"].map(provider_count)
            x["AttendingPhysician"]=x["AttendingPhysician"].map(att_phy_count)
            reimb_amount=x["IPAnnualReimbursementAmt"]+x["OPAnnualReimbursementAmt"]
            deduct_amt=x["IPAnnualDeductibleAmt"]+x["OPAnnualDeductibleAmt"]
            x["amount_to_reimbursed"]=int(reimb_amount)-int(deduct_amt)

            count_of_physician=x[['AttendingPhysician','OperatingPhysician','OtherPhysician']].fillna(0).values
            count=[]
            for i in range(len(count_of_physician)):
                count.append(np.count_nonzero(count_of_physician[i]))
            x["Physician_count"]=count

            for i in d_codes:
                x["D_"+str(i)]=np.where(x["ClmDiagnosisCode_1"]==i,1,0)
                for j in range(2,11):
                    x["D_"+str(i)]=np.where(x["ClmDiagnosisCode_"+str(j)]==i,1,np.where(x['D_'+str(i)]==1,1,0))
                x['D_'+str(i)] = np.where(x['DiagnosisGroupCode']==i,1,np.where(x['D_'+str(i)]==1,1,0 ))
                x['D_'+str(i)] = np.where(x['ClmAdmitDiagnosisCode']==i,1,np.where(x['D_'+str(i)]==1,1,0 ))

            for i in p_codes:
                x["P_"+str(i)]=np.where(x["ClmProcedureCode_1"]==i,1,0)
                for j in range(2,4):
                    x["P_"+str(i)]=np.where(x["ClmProcedureCode_"+str(j)]==i,1,np.where(x['P_'+str(i)]==1,1,0))


            x.drop(columns=["ClmProcedureCode_1","ClmProcedureCode_2","ClmProcedureCode_3","OtherPhysician"
                        ,"OperatingPhysician","IPAnnualReimbursementAmt","OPAnnualReimbursementAmt","IPAnnualDeductibleAmt"
                        ,"OPAnnualDeductibleAmt","ClmProcedureCode_4","ClmDiagnosisCode_1","ClmDiagnosisCode_2",
                         "ClmDiagnosisCode_3","ClmDiagnosisCode_4","ClmDiagnosisCode_5","ClmDiagnosisCode_6",
                        "ClmDiagnosisCode_7","ClmDiagnosisCode_8","ClmDiagnosisCode_9","ClmDiagnosisCode_10",
                        "DiagnosisGroupCode","ClmAdmitDiagnosisCode"]
                          ,inplace=True,axis=1)
      
            for i in sccale_cols:
                sc=load('scaling/'+i+'_std_scaler.bin')
                x[i]=sc.transform(x[i].values.reshape(-1,1))

            return x

        x=data_for_test(df)
        print(x.shape)
        prediction=model.predict(x)


        if prediction[0]==1:
            return render_template('deployment.html',prediction_texts="FRAUD CLAIM")
        else:
            return render_template('deployment.html',prediction_text="NON FRAUD CLAIM")
     
       

if __name__=="__main__":
    app.run(debug=True)

