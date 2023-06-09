import streamlit
import pickle
import numpy

prediction_model = pickle.load(open('ml_regression_model.pkl','rb'))


def prediction_gen(input_data):
    prediction = prediction_model.predict(input_data)
    return {"Temperature": prediction[0][0]}

def main():
    
    #title
    streamlit.title('Weather Prediction')
   
    #inputdataset
    humidity_var     = streamlit.text_input('Humidity (in Percent)')
    wind_speed_var   = streamlit.text_input('Wind Speed (in Kmph)')
    meanpressure_var = streamlit.text_input('Pressure (in mbar)')
    
    result = ''
    
    #creating a button
    if streamlit.button('Predict Temperature'):
        result = prediction_gen(numpy.array([[humidity_var,wind_speed_var,meanpressure_var]]))
        
    streamlit.success(result)

if __name__ == '__main__':
    main()
