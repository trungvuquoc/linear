import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

df = pd.read_csv("credit access.csv", encoding='latin-1')

st.title("Hồi quy tuyến tính")
st.write("## Dự báo giá trị vay vốn của nông hộ")

uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("data.csv", index = False)

X = df.drop(columns=['giatri'])
y = df['giatri']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 12)

model = LinearRegression()

model.fit(X_train, y_train)

yhat_test = model.predict(X_test)


score_train=model.score(X_train, y_train)
score_test=model.score(X_test, y_test)


mse=mean_squared_error(y_test, yhat_test)
rmse=mean_squared_error(y_test, yhat_test, squared=False)
mae=mean_absolute_error(y_test, yhat_test)


menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của mô hình':    
    st.subheader("Mục tiêu của mô hình")
    st.write("""
    ###### Mô hình được xây dựng để dự báo giá trị vay vốn của nông hộ dựa trên các biến đặc điểm chủ hộ, điều kiện của nông hộ.
    """)  
    st.write("""###### Mô hình sử dụng thuật toán LinearRegression""")
    st.image("LSM.png")
    st.image("LSM_1.png")

elif choice == 'Xây dựng mô hình':
    st.subheader("Xây dựng mô hình")
    st.write("##### 1. Hiển thị dữ liệu")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  
    
    st.write("##### 2. Trực quan hóa dữ liệu")
    u=st.text_input('Nhập biến muốn vẽ vào đây')
    fig1 = sns.regplot(data=df, x=u, y='giatri')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model...")
    
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("MSE:"+str(round(mse,2)))
    st.code("RMSE:"+str(round(rmse,2)))
    st.code("MAE:"+str(round(mae,2)))

    
elif choice == 'Sử dụng mô hình để dự báo':
    st.subheader("Sử dụng mô hình để dự báo")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            # st.write(lines.columns)
            flag = True       
    if type=="Input":        
        git = st.number_input('Insert giatri')
        DT = st.number_input('Insert DT')
        TN = st.number_input('Insert TN')
        SPT = st.number_input('Insert SPT')
        GTC = st.number_input('Insert GTC')
        GD = st.number_input('Insert GD')
        TCH = st.number_input('Insert TCH')
        GT = st.number_input('Insert GT')
        DV = st.number_input('Insert DV')
        VPCT = st.number_input('Insert VPCT')
        LS = st.number_input('Insert LS')
        lines={'giatri':[git],'DT':[DT],'TN':[TN],'SPT':[SPT],'GTC':[GTC],'GD':[GD],'TCH':[TCH],'GT':[GT],'DV':[DV],'VPCT':[VPCT],'LS':[LS]}
        lines=pd.DataFrame(lines)
        st.dataframe(lines)
        flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            X_1 = lines.drop(columns=['giatri'])   
            y_pred_new = model.predict(X_1)       
            st.code("giá trị dự báo: " + str(y_pred_new))
