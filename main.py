import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

messages = []

print("คุยกับ AI ได้เลย! พิมพ์ 'exit' เพื่อออก")

while True:
    user_input = input("คุณ: ")
    
    if user_input == "exit":
        print("ลาก่อนครับ!")
        break
    
    messages.append(HumanMessage(content=user_input))
    
    response = llm.invoke(messages)
    
    messages.append(AIMessage(content=response.content))
    
    print(f"AI: {response.content}\n")