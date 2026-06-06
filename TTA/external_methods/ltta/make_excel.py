import pandas as pd
import sys

AUX_NUM = 'IMG_D2_K3_FIX_SE_RAW_EP10_Red32_POS0_TOP.csv'
TARGET_DIR = 'csv_red32_single_p7'
# File paths
# gaussian_noise_IMG_D2_K3_FIX_SE_RAW_EP10_Red16_POS0.csv
file_paths = [
    '/gaussian_noise_' + AUX_NUM,
    '/shot_noise_' + AUX_NUM,
    '/impulse_noise_' + AUX_NUM,
    '/defocus_blur_' + AUX_NUM,
    '/glass_blur_' + AUX_NUM,
    '/motion_blur_' + AUX_NUM,
    '/zoom_blur_' + AUX_NUM,
    '/snow_' + AUX_NUM,
    '/frost_' + AUX_NUM,
    '/fog_' + AUX_NUM,
    '/brightness_' + AUX_NUM,
    '/contrast_' + AUX_NUM,
    '/elastic_transform_' + AUX_NUM,
    '/pixelate_' + AUX_NUM,
    '/jpeg_compression_' + AUX_NUM,
]

# Load data from each file and create a single Excel file with multiple sheets

excel_writer = pd.ExcelWriter(f'{TARGET_DIR}/'+sys.argv[1]+".xlsx")
print(f'{TARGET_DIR}/'+sys.argv[1]+".xlsx")

for file_path in file_paths:
    # Extracting sheet name from the file path
    file_path = TARGET_DIR + file_path
    sheet_name = file_path.split("/")[-1].split("_")[0]

    # Read CSV file
    df = pd.read_csv(file_path)

    # Write to a sheet in the Excel file
    df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

# Save the Excel file
excel_writer.save()