import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

messages = [
    {
        "role": "system",
        "content": "คุณเป็น AI assistant ชื่อ บี ช่วยสอน Python และ AI development ตอบภาษาไทย กระชับและตรงประเด็น"
    }
]

print("คุยกับ บี ได้เลย! พิมพ์ 'exit' เพื่อออก")

while True:
    user_input = input("คุณ: ")
    
    if user_input == "exit":
        print("ลาก่อนครับ!")
        break
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    ai_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": ai_reply})
    
    print(f"บี: {ai_reply}\n")