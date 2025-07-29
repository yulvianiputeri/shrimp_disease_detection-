import os
import csv
from pathlib import Path

def buat_dataset_csv():
    """Membuat file CSV dari struktur folder dataset gambar udang"""
    folder_data = 'data_udang'
    file_csv = 'shrimp_dataset.csv'
    
    if not os.path.exists(folder_data):
        print(f"Error: Folder '{folder_data}' tidak ditemukan!")
        return False
    
    mapping_folder = {
        '1. Healthy': 'Healthy',
        '2. BG': 'BG',
        '3. WSSV': 'WSSV', 
        '4. WSSV_BG': 'WSSV_BG'
    }
    
    print(f"Membuat dataset CSV dari folder: {folder_data}")
    
    try:
        with open(file_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['image_path', 'label'])
            
            total_gambar = 0
            
            for nama_folder in os.listdir(folder_data):
                path_folder = os.path.join(folder_data, nama_folder)
                
                if not os.path.isdir(path_folder):
                    continue
                    
                label = mapping_folder.get(nama_folder, nama_folder.split('. ', 1)[-1])
                
                gambar_dalam_folder = 0
                
                for nama_file in os.listdir(path_folder):

                    if nama_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        path_gambar = os.path.join(path_folder, nama_file)
                        
                        if os.path.isfile(path_gambar):
                            writer.writerow([path_gambar, label])
                            gambar_dalam_folder += 1
                            total_gambar += 1
                
                print(f"Folder '{nama_folder}' -> Label: '{label}' -> {gambar_dalam_folder} gambar")
        
        print(f"\nDataset CSV berhasil dibuat: {file_csv}")
        print(f"Total gambar yang diproses: {total_gambar}")
        return True
        
    except Exception as e:
        print(f"Error saat membuat CSV: {e}")
        return False

def verifikasi_dataset():
    """Verifikasi dataset yang telah dibuat"""
    file_csv = 'shrimp_dataset.csv'
    
    if not os.path.exists(file_csv):
        print("File CSV belum dibuat!")
        return
    
    try:
        import pandas as pd
        df = pd.read_csv(file_csv)
        
        print(f"\nVerifikasi dataset:")
        print(f"Total baris: {len(df)}")
        print(f"Distribusi label:")
        print(df['label'].value_counts())
        
        file_hilang = 0
        for path in df['image_path']:
            if not os.path.exists(path):
                file_hilang += 1
        
        if file_hilang > 0:
            print(f"Peringatan: {file_hilang} file gambar tidak ditemukan!")
        else:
            print("Semua file gambar tersedia!")
            
    except Exception as e:
        print(f"Error saat verifikasi: {e}")

if __name__ == "__main__":
    if buat_dataset_csv():
        verifikasi_dataset()
    else:
        print("Gagal membuat dataset CSV!")