import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

@tool
def calculate(expression: str) -> str:
    """คำนวณสมการคณิตศาสตร์ เช่น 2+2, 10*5, 100/4"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "คำนวณไม่ได้ครับ"

@tool
def get_date_info(date_str: str) -> str:
    """บอกข้อมูลวันที่ รับ format dd/mm/yyyy หรือ 'today' สำหรับวันนี้"""
    try:
        from datetime import datetime
        days = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        months = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
                  "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]

        if date_str.lower() == "today":
            date = datetime.now()
        else:
            date = datetime.strptime(date_str, "%d/%m/%Y")

        today = datetime.now()
        diff = (date - today).days

        if diff == 0:
            relation = "คือวันนี้"
        elif diff > 0:
            relation = f"อีก {diff} วันข้างหน้า"
        else:
            relation = f"ผ่านมาแล้ว {abs(diff)} วัน"

        return (
            f"วันที่ {date.day} {months[date.month-1]} {date.year} "
            f"เป็นวัน{days[date.weekday()]} — {relation}"
        )
    except:
        return "รูปแบบวันที่ไม่ถูกต้อง กรุณาใส่แบบ dd/mm/yyyy"
    
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """แปลงสกุลเงิน รองรับ THB, USD, EUR, JPY, GBP"""
    rates = {
        "THB": 1,
        "USD": 0.028,
        "EUR": 0.026,
        "JPY": 4.2,
        "GBP": 0.022,
        "CNY":0.20,
        "KRW": 37.5,
        "AUD": 0.043,
        "SGD": 0.038,
        "MYR": 0.13
    }
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency not in rates or to_currency not in rates:
        return "ไม่รองรับสกุลเงินนี้ครับ รองรับแค่ THB, USD, EUR, JPY, GBP"
    result = amount / rates[from_currency] * rates[to_currency]
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"

@tool
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """แปลงหน่วย รองรับ km/miles, kg/lbs, celsius/fahrenheit"""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x * 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    }
    key = (from_unit, to_unit)
    if key not in conversions:
        return f"ไม่รองรับการแปลง {from_unit} → {to_unit} ครับ"
    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"

tools = [calculate, get_date_info, convert_currency, convert_unit]
llm_with_tools = llm.bind_tools(tools)

messages = [
    SystemMessage(content="คุณเป็น AI assistant ที่มีเครื่องมือคำนวณ ตอบภาษาไทย")
]

print("คุยกับ AI ได้เลย! พิมพ์ 'exit' เพื่อออก")

while True:
    user_input = input("คุณ: ")
    
    if user_input == "exit":
        print("ลาก่อนครับ!")
        break
    
    messages.append(HumanMessage(content=user_input))
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            
            if tool_name == "calculate":
                tool_result = calculate.invoke(tool_call["args"])
            elif tool_name == "get_date_info":
                tool_result = get_date_info.invoke(tool_call["args"])
            elif tool_name ==   "convert_currency":
                 tool_result = convert_currency.invoke(tool_call["args"])
            elif tool_name == "convert_unit":
                tool_result = convert_unit.invoke(tool_call["args"])
            else:
                tool_result = "ไม่พบ tool นี้ครับ"
            
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            ))
        
        final_response = llm_with_tools.invoke(messages)
        messages.append(final_response)
        print(f"AI: {final_response.content}\n")
    else:
        print(f"AI: {response.content}\n")
    

