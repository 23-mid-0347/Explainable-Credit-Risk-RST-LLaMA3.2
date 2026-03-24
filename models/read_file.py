import pickle

path = r'D:\Coding\6th sem\Soft Computing\Credit Risk Assessment\models\rf_model.pkl'

try:
    with open(path, 'rb') as f:
        # Adding encoding='latin1' helps map old strings to new Python 3 strings
        data = pickle.load(f, encoding='latin1')
        
    print("--- Success! ---")
    print(f"Object Type: {type(data)}")
    
    # If it's a Scikit-Learn model, print the parameters
    if hasattr(data, 'get_params'):
        print("Model Parameters:", data.get_params())
        
except Exception as e:
    print(f"Still failing. Error: {e}")