from glob import glob
from reading_data import read_data
from resample_data import resample_data, export_data



if __name__ == "__main__":
    files = glob("data/raw/*.csv")
    acc_df, gyro_df = read_data(files)
    data = resample_data(files)
    
    output_path = "data/partially processed/resampled_data.csv"
    export_data(data, output_path)
    
    