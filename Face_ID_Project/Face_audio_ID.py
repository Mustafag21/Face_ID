import cv2
import mysql.connector
import numpy as np
import time
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
from deepface import DeepFace


veritabani = mysql.connector.connect(
    host="localhost",
    user="root",
    password="mustafa21",
    database="biometri_db"
)
imlec = veritabani.cursor()

imlec.execute("SELECT * FROM fotograflar")
rows = imlec.fetchall()
yuzler_listesi = []
for row in rows:
    yuzler_listesi.append({
        "id": row[0],
        "dosya_yolu": row[1],
        "ad": row[2],
        "soyad": row[3],
        "kayit_tarihi": row[4],
        "kullanici_id": row[5],
        "face_features": np.frombuffer(row[6], dtype=np.float64),
        "ses_dosya_yolu": row[7]
    })

def record_audio(duration=5):
    fs = 44100
    print("Ses kaydediliyor...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Ses kaydı tamamlandı.")
    return myrecording, fs

def extract_mfcc(audio_data, sr=44100):
    y = audio_data.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
    return np.mean(mfcc.T, axis=0)

def match_audio_features(database_audio_path, user_audio, sr=44100):
    try:
        database_audio, _ = librosa.load(database_audio_path, sr=sr)
        database_mfcc = extract_mfcc(database_audio, sr)
        user_mfcc = extract_mfcc(user_audio, sr)
        similarity = np.dot(database_mfcc, user_mfcc) / (np.linalg.norm(database_mfcc) * np.linalg.norm(user_mfcc))
        return similarity
    except Exception as e:
        print(f"Error in audio matching: {e}")
        return -1

def get_face_encoding(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(img_path=rgb_frame, model_name='Facenet', enforce_detection=False)
        if len(result) > 0:
            return result[0]['embedding']
    except Exception as e:
        print(f"Error in face encoding: {e}")
    return None

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

while True:
    gecerli_zaman = time.localtime()
    biçimli_zaman = time.strftime("%Y-%m-%d", gecerli_zaman)
    biçimli_zaman2 = time.strftime("%H:%M:%S")
    print("\nMERHABA! Bugünün Tarihi:", biçimli_zaman, "Ve Saat:", biçimli_zaman2)
    print(
        "\n VERİ TABANINA YÜZ KAYDETMEK VE SES KAYDETMEK İSTİYORSANIZ 1 TUŞUNA BASINIZ \n"
        "\n VERİ TABANINDAN SES İLE KİŞİ SORGUSU YAPILMASI İÇİN 2 TUŞUNA BASINIZ\n"
    )
    cevap = int(input("Lütfen seçiminizi giriniz: "))

    if cevap == 1:
        camera = cv2.VideoCapture(0)
        yuz_detektoru = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.resize(frame, (900, 500))
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = yuz_detektoru.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (85, 255, 0), 3)
            cv2.imshow('Yuz Tespiti', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('k'):
                gecerli_zaman = time.strftime("%Y-%m-%d", time.localtime())
                kullanici_id = 1
                log_face_encoding = get_face_encoding(frame)
                if log_face_encoding is not None:
                    eslesme = False
                    for yuz in yuzler_listesi:
                        yuz_encoding = yuz["face_features"]
                        similarity = np.dot(log_face_encoding, yuz_encoding) / (
                                    np.linalg.norm(log_face_encoding) * np.linalg.norm(yuz_encoding))
                        if similarity > 0.5:
                            eslesme = True
                            ad = yuz["ad"]
                            soyad = yuz["soyad"]
                            print('\n\nEşleşen Kişi: {} {}'.format(ad, soyad))
                            break
                    if not eslesme:
                        ad = input("\nİsim giriniz: ")
                        soyad = input("Soyisim giriniz: ")
                        create_directory("yuzler")
                        create_directory("sesler")

                        dosya_yolu = f"yuzler/{ad}_{soyad}_{int(time.time())}.jpg"
                        cv2.imwrite(dosya_yolu, frame)
                        ses_dosya_yolu = f"sesler/{ad}_{soyad}_{int(time.time())}.wav"
                        user_audio, fs = record_audio()
                        user_audio_int16 = (user_audio * 32767).astype(np.int16)
                        wav.write(ses_dosya_yolu, fs, user_audio_int16)
                        imlec.execute(
                            "INSERT INTO fotograflar (ad, soyad, dosya_yolu, kayit_tarihi, kullanici_id, face_features, ses_dosya_yolu) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (ad, soyad, dosya_yolu, gecerli_zaman, kullanici_id, np.array(log_face_encoding).tobytes(),
                             ses_dosya_yolu)
                        )
                        veritabani.commit()
                        print("Yüz ve ses kaydı işlemi tamamlandı.")
                        time.sleep(3)
                camera.release()
                cv2.destroyAllWindows()
                break

    elif cevap == 2:
        user_audio, fs = record_audio()

        matched_person = None
        highest_similarity = -1
        for person in yuzler_listesi:
            database_audio_path = person["ses_dosya_yolu"]
            similarity = match_audio_features(database_audio_path, user_audio.flatten(), sr=fs)
            if similarity > highest_similarity:
                highest_similarity = similarity
                matched_person = person

        if matched_person and highest_similarity > 0.5:
            ad = matched_person["ad"]
            soyad = matched_person["soyad"]
            kayıt_tarihi = matched_person["kayit_tarihi"]
            print("Eşleşen kişinin bilgileri:")
            print('\n\nEşleşen Kişi: {} {}'.format(ad, soyad))
            print("Kayıt Tarihi:", kayıt_tarihi, "\n")
            if "dosya_yolu" in matched_person:
                dosya_yolu = matched_person["dosya_yolu"]
                image = cv2.imread(dosya_yolu)
                cv2.imshow(f"PERSON ID: {ad, soyad}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Eşleşen kişi bulunamadı. Lütfen sesi doğru şekilde kaydedin ve tekrar deneyin.")


