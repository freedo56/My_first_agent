import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from datetime import datetime

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

@tool
def calculate(expression: str) -> str:
    """คำนวณสมการคณิตศาสตร์"""
    try:
        return str(eval(expression))
    except:
        return "คำนวณไม่ได้ครับ"

@tool
def get_date_info(date_str: str) -> str:
    """บอกข้อมูลวันที่ รับ dd/mm/yyyy หรือ today"""
    try:
        days = ["จันทร์","อังคาร","พุธ","พฤหัสบดี","ศุกร์","เสาร์","อาทิตย์"]
        date = datetime.now() if date_str.lower() == "today" else datetime.strptime(date_str, "%d/%m/%Y")
        diff = (date - datetime.now()).days
        relation = "คือวันนี้" if diff == 0 else f"อีก {diff} วัน" if diff > 0 else f"ผ่านมา {abs(diff)} วันแล้ว"
        return f"{date.strftime('%d/%m/%Y')} วัน{days[date.weekday()]} — {relation}"
    except:
        return "รูปแบบวันที่ไม่ถูกต้องครับ"

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """แปลงสกุลเงิน THB USD EUR JPY GBP"""
    rates = {"THB":1,"USD":0.028,"EUR":0.026,"JPY":4.2,"GBP":0.022}
    f, t = from_currency.upper(), to_currency.upper()
    if f not in rates or t not in rates:
        return "ไม่รองรับสกุลเงินนี้ครับ"
    return f"{amount} {f} = {amount/rates[f]*rates[t]:.2f} {t}"

@tool
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """แปลงหน่วย km/miles, kg/lbs, celsius/fahrenheit"""
    conv = {("km","miles"):lambda x:x*0.621371,("miles","km"):lambda x:x*1.60934,
            ("kg","lbs"):lambda x:x*2.20462,("lbs","kg"):lambda x:x*0.453592,
            ("celsius","fahrenheit"):lambda x:x*9/5+32,("fahrenheit","celsius"):lambda x:(x-32)*5/9}
    key = (from_unit.lower(), to_unit.lower())
    return f"{value} {from_unit} = {conv[key](value):.4f} {to_unit}" if key in conv else "ไม่รองรับการแปลงนี้ครับ"

tools = [calculate, get_date_info, convert_currency, convert_unit]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

st.title("🤖 AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="คุณเป็น AI assistant ชื่อ บี ตอบภาษาไทย กระชับตรงประเด็น")
    ]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)

if prompt := st.chat_input("พิมพ์ข้อความ..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    response = llm_with_tools.invoke(st.session_state.messages)
    st.session_state.messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            st.session_state.messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        response = llm_with_tools.invoke(st.session_state.messages)
        st.session_state.messages.append(response)

    with st.chat_message("assistant"):
        st.write(response.content)
        