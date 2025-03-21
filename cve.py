import os
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool  # 改用结构化工具类
import pandas as pd
from typing import Dict, Any

# 数据加载模块
def load_cve_data(file_path: str) -> pd.DataFrame:
    """加载并预处理CVE数据"""
    df = pd.read_excel(file_path)
    df["CVE编号"] = df["CVE编号"].astype("string")
    return df

# 工具定义模块（使用新式工具定义）
def search_references_url(cve_id: str) -> str:
    """根据CVE编号查询对应的References链接"""
    try:
        cve_id = cve_id.strip().upper()
        result = CVE_DF[CVE_DF["CVE编号"] == cve_id]
        return result["References"].values[0] if not result.empty else "未找到相关信息"
    except Exception as e:
        return f"查询失败：{str(e)}"

def get_cve_descriptions(cve_id: str) -> str:
    """根据CVE编号查询漏洞描述信息"""
    try:
        cve_id = cve_id.strip().upper()
        result = CVE_DF[CVE_DF["CVE编号"] == cve_id]
        return result["漏洞描述"].values[0] if not result.empty else "未找到相关信息"
    except Exception as e:
        return f"查询失败：{str(e)}"

# 创建结构化工具对象（关键修改）
tools = [
    StructuredTool.from_function(
        func=search_references_url,
        name="search_references_url",
        description="根据CVE编号查询References链接"
    ),
    StructuredTool.from_function(
        func=get_cve_descriptions,
        name="get_cve_descriptions",
        description="根据CVE编号查询漏洞描述"
    )
]

if __name__ == "__main__":
    # 初始化配置
    CVE_DF = load_cve_data("test_one.xlsx")
    
    # 安全获取API密钥（建议使用环境变量）
    api_key = "sk-u3ph8f8eivL6YnttJdgwM8imj3lPE0MPJs5SrF1nTkqJZkXs"
    
    # 初始化模型（兼容性配置）
    model_bind = ChatOpenAI(
        model="gpt-4",
        base_url="https://xiaoai.plus/v1",
        api_key=api_key,
        temperature=0.5
    ).bind(system="你是一名专业的网络安全分析师")
    
    # 创建Agent（关键修改）
    agent = create_react_agent(
        model=model_bind,
        tools=tools,

    )
    
    # 执行查询
    try:
        response = agent.invoke({
            "messages": [
                SystemMessage(content="你只能使用提供的工具查询CVE信息"),
                HumanMessage(content="请查询CVE-2022-4938的漏洞描述和References链接")
            ]
        })
        Last_messages=response.get("messages")[-1]
        print("最终结果：", Last_messages.content)
        
    except Exception as e:
        print(f"执行错误：{str(e)}")