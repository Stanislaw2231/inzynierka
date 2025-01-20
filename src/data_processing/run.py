from glob import glob
from reading_data import read_data
from resample_data import resample_data, export_data
from detecting_outliers import LOF_outliers


if __name__ == "__main__":
    files = glob("data/raw/*.csv")
    acc_df, gyro_df = read_data(files)
    data = resample_data(files)
    
    output_path = "data/partially processed/resampled_data.csv"
    export_data(data, output_path)
    
    #outlier detection and removal
    data = LOF_outliers(data)
    export_data(data, "data/partially processed/resampled_data_no_outliers.csv")
    
    