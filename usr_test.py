import cv2
import sqlite3
from ultralytics import YOLO
from datetime import datetime

# 定義全域變數用於記錄上一次記錄的秒數
last_recorded_second = -1

def create_table():
    # 連接到資料庫或建立新的資料庫
    conn = sqlite3.connect('count_person.db')
    cursor = conn.cursor()
    
    # 創建 count_person 資料表
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS count_person (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        formatted_time DATETIME NOT NULL,
        person_count INTEGER NOT NULL
    );
    '''
    cursor.execute(create_table_query)
    
    # 提交並關閉資料庫連接
    conn.commit()
    conn.close()

def insert_data(person_count, formatted_time):
    conn = sqlite3.connect('count_person.db')
    cursor = conn.cursor()
    
    # 插入資料
    insert_query = '''
    INSERT INTO count_person (formatted_time, person_count)
    VALUES (?, ?);
    '''
    cursor.execute(insert_query, (formatted_time, person_count))
    
    conn.commit()
    conn.close()

def inference(modelName):
    global last_recorded_second

    # Load the YOLOv8 model
    model = YOLO(modelName)

    # 開啟 RTSP 串流
    vidCap = cv2.VideoCapture('rtsp://fclinlab:fcu@216818@192.168.0.128/stream1')

    # Loop through the video frames
    while vidCap.isOpened():
        # Read a frame from the video
        success, frame = vidCap.read()
        if success:
            results = model.predict(frame, stream=True, classes=0)
            
            for idx in results:
                annotated_frame = idx.plot()
                annotated_box = idx.boxes
                count_person = idx.boxes.shape[0]

            # 輸出內容看這邊：https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
            # print(annotated_box)

            # 取得當前時間戳記
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

            if current_time.second % 5 == 0 and current_time.second != last_recorded_second:
                insert_data(count_person, formatted_time)
                last_recorded_second = current_time.second
            
            cv2.putText(annotated_frame, str(count_person), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    vidCap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 建立資料表
    create_table()
    # 執行辨識
    inference('yolov8l.pt')