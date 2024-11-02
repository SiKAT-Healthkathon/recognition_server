import os
from dotenv import load_dotenv
import face_recognition
import asyncio
import websockets
import json, io, base64
from supabase import create_client, Client

load_dotenv()

# Inisialisasi Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

async def websocket_handler(websocket):
    try:
        async for message in websocket:
            response = await recognize_face(message)
            await websocket.send(json.dumps(response))
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")

async def fetch_all_user_images():
    try:
        # Ambil semua data pengguna dari database
        response = supabase.from_("users").select("nik, photo").execute()
        return response.data
    except Exception as e:
        print(f"Error fetching user images: {str(e)}")
        return []

async def recognize_face(image_bytes):
    try:
        # Decode dan proses gambar dari kamera
        unknown_picture = face_recognition.load_image_file(io.BytesIO(image_bytes))
        unknown_face_encodings = face_recognition.face_encodings(unknown_picture)

        if len(unknown_face_encodings) == 0:
            return {"status": 0, "message": "No Face Detected", "data": None}
        
        unknown_face_encoding = unknown_face_encodings[0]

        # Ambil semua data pengguna
        users_data = await fetch_all_user_images()

        # Iterasi melalui semua pengguna untuk mencocokkan wajah
        for user in users_data:
            nik = user["nik"]
            user_photo_base64 = user["photo"]

            if user_photo_base64.startswith("data:image/"):
                user_photo_base64 = user_photo_base64.split(",")[1]

            # Decode gambar pengguna dari base64 ke bytes
            user_image_bytes = base64.b64decode(user_photo_base64)
            known_image = face_recognition.load_image_file(io.BytesIO(user_image_bytes))
            known_face_encodings = face_recognition.face_encodings(known_image)

            if len(known_face_encodings) == 0:
                print(f"Warning: No face encoding found for nik {nik}")
                continue

            known_face_encoding = known_face_encodings[0]

            # Pencocokan wajah
            results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)
            if results[0]:  # Jika cocok
                return {"status": 2, "message": "Recognition successful", "data": nik}

        return {"status": 1, "message": "Face not recognized", "data": None}

    except Exception as e:
        return {"status": 1, "message": str(e)}

async def main():
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever

if __name__ == '__main__':
    asyncio.run(main())

