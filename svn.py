import os
import re
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
import logging
from openai import OpenAI
import re
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== 数据加载模块 ==================
CVE_DF = pd.read_excel("test_one.xlsx")
CVE_DF["CVE编号"] = CVE_DF["CVE编号"].astype("string")

# ================== 核心分析模块 ==================
class CVEAnalyzer:
    """增强版CVE分析器（集成非Agent代码优势）"""
    def __init__(self):
        self.current_cve: Optional[Dict] = None
        self.svn_logs: List[Dict] = []
        self.candidate_commits: List[Dict] = []
        self.attempt_count: int = 0
        self.max_attempts: int = 5
        self.WINDOW_SIZE = 15  # 扩大搜索范围

    def load_cve_data(self, cve_id: str) -> Optional[Dict]:
        """加载CVE基础数据"""
        try:
            cve_id = cve_id.upper().strip()
            record = CVE_DF[CVE_DF["CVE编号"] == cve_id].iloc[0]
            self.current_cve = {
                "id": cve_id,
                "description": record["漏洞描述"],
                "affected_versions": record["影响版本"],
                "package": record["packagename"]
            }
            logger.info(f"已加载CVE数据：{cve_id}")
            return self.current_cve
        except Exception as e:
            logger.error(f"CVE数据加载失败：{str(e)}")
            return None

    def parse_version_condition(self):
     version_str = self.current_cve["affected_versions"]
    # 匹配 "Affect 4.3.1 < 4.3.1" 中的有效版本
     versions = re.findall(r'\d+\.\d+\.\d+', str(version_str))
     if len(versions) >= 2:
        return (versions[1], False)  # (target_version=4.3.1, allow_equal=False)
     return None, None

    def fetch_svn_logs(self) -> Optional[List[Dict]]:
        """获取并解析SVN日志"""
        try:
            result = subprocess.run(
                ['svn', 'log', '--xml', 
                 f'https://plugins.svn.wordpress.org/{self.current_cve["package"]}'],
                capture_output=True, check=True, timeout=30
            )
            xml_output = self._decode_svn_output(result)
            root = ET.fromstring(xml_output)
            
            self.svn_logs = [{
                'revision': entry.get('revision'),
                'msg': entry.find('msg').text,
                'used': False
            } for entry in root.findall('logentry')]
            
            logger.info(f"已获取{len(self.svn_logs)}条SVN日志")
            return self.svn_logs[::-1]  # 倒序排列
        except subprocess.CalledProcessError as e:
            logger.error(f"SVN命令执行失败：{self._decode_svn_output(e)}")
            return None
        except ET.ParseError:
            logger.error("SVN日志XML解析失败")
            return None

    def locate_candidate_commits(self, version: str, allow_equal: bool):
    # 通过SVN tags定位基础提交
     base_commit = self._find_base_commit(version)
     sorted_logs = self.svn_logs[::-1]  # 时间正序
    
    # 找到base_commit在日志中的位置
     target_index = next(i for i, log in enumerate(sorted_logs)
                       if log['revision'] == base_commit['commit_id'])
    
    # 向前扩展窗口（旧提交方向）
     start = max(0, target_index - self.WINDOW_SIZE)
     candidates = sorted_logs[start:target_index]
    
    # 添加关键提交验证
     if base_commit['commit_id'] not in [c['revision'] for c in candidates]:
        candidates.append(base_commit)
    
     return candidates
 
    def get_next_diff(self) -> Optional[Tuple[str, str]]:
        """带优先级的差异获取"""
        sorted_commits = sorted(
        self.candidate_commits,
        key=lambda x: int(x['revision']),
        reverse=True  # 优先分析较新提交
     )
        for commit in sorted_commits:
         if diff := self._generate_diff(commit):
              return diff
        return None

    def _generate_diff(self, commit: Dict) -> Optional[Tuple[str, str]]:
        """生成并过滤差异"""
        try:
            prev_rev = str(int(commit['revision']) - 1)
            result = subprocess.run(
                ['svn', 'diff', '-r', f"{prev_rev}:{commit['revision']}",
                 f"https://plugins.svn.wordpress.org/{self.current_cve['package']}"],
                capture_output=True, check=True, timeout=30
            )
            commit['used'] = True
            self.attempt_count += 1
            filtered_diff = self._filter_diff(result.stdout.decode('utf-8'))
            logger.info(f"获取到差异：{commit['revision']}")
            return (commit['revision'], filtered_diff)
        except subprocess.TimeoutExpired:
            logger.error("命令执行超时")
            return None
        except Exception as e:
            logger.error(f"差异获取失败：{str(e)}")
            return None

    def _filter_diff(self, diff_text: str) -> str:
     """关键路径过滤优化"""
     filtered = []
     current_file = None
     is_relevant = False
    
     for line in diff_text.split('\n'):
        if line.startswith('Index:'):
            current_file = line.split()[-1]
            is_relevant = any(
                path in current_file 
                for path in ('trunk/admin/', 'trunk/includes/')
            ) and current_file.endswith(('.php', '.js'))
        if is_relevant:
            filtered.append(line)
     return '\n'.join(filtered)

    def _match_version(self, text: str, target_version: str) -> bool:
        """智能版本匹配"""
        def process_version(v: str) -> str:
            parts = v.split('.')
            while len(parts) > 1 and parts[-1] == '0':
                parts.pop()
            return '.'.join(parts)
        
        found_versions = re.findall(r'\b\d+(?:\.\d+)*\b', text)
        processed_target = process_version(target_version)
        return any(process_version(v) == processed_target for v in found_versions)

    def _compare_versions(self, v1: str, v2: str) -> int:
        """标准版本比较"""
        v1_parts = list(map(int, v1.split('.')))
        v2_parts = list(map(int, v2.split('.')))
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts += [0] * (max_len - len(v1_parts))
        v2_parts += [0] * (max_len - len(v2_parts))
        
        for a, b in zip(v1_parts, v2_parts):
            if a < b: return -1
            elif a > b: return 1
        return 0

    def _fetch_svn_tags(self) -> Optional[str]:
        """获取SVN tags数据"""
        try:
            result = subprocess.run(
                ['svn', 'list', '--xml', 
                 f'https://plugins.svn.wordpress.org/{self.current_cve["package"]}/tags'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return self._decode_svn_output(result)
        except Exception as e:
            logger.error(f"获取tags失败：{str(e)}")
            return None

    def _parse_svn_tags(self, xml_data: str) -> List[Dict]:
        """解析tags数据"""
        try:
            root = ET.fromstring(xml_data)
            return [{
                'commit_id': entry.find('commit').get('revision'),
                'version': entry.find('name').text
            } for entry in root.findall('.//entry') if entry.find('commit') and entry.find('name')]
        except ET.ParseError:
            logger.error("Tags XML解析失败")
            return []

    def _find_max_commit(self, tags: List[Dict], target_version: str, allow_equal: bool) -> Optional[Dict]:
        """查找最大匹配提交"""
        best_match = None
        for tag in tags:
            comparison = self._compare_versions(tag['version'], target_version)
            if (comparison < 0) or (allow_equal and comparison == 0):
                if not best_match or self._compare_versions(tag['version'], best_match['version']) > 0:
                    best_match = tag
        return best_match

    def _decode_svn_output(self, result: subprocess.CompletedProcess) -> str:
        """多编码格式解码"""
        encodings = ['utf-8', 'gbk', 'latin-1']
        for encoding in encodings:
            try:
                return result.stdout.decode(encoding)
            except UnicodeDecodeError:
                continue
        logger.error("SVN输出解码失败")
        return ""

# ================== 工具和模型配置 ==================
cve_analyzer = CVEAnalyzer()

tools = [
    StructuredTool.from_function(
        func=cve_analyzer.load_cve_data,
        name="load_cve_data",
        description="根据CVE ID加载漏洞基本信息，参数格式：CVE编号（如CVE-2022-3995）"
    ),
    StructuredTool.from_function(
        func=cve_analyzer.parse_version_condition,
        name="parse_version",
        description="解析影响版本条件，返回（版本号, 是否允许等于）"
    ),
    StructuredTool.from_function(
        func=cve_analyzer.fetch_svn_logs,
        name="fetch_svn_logs",
        description="获取SVN提交历史，返回倒序日志列表"
    ),
    StructuredTool.from_function(
        func=cve_analyzer.locate_candidate_commits,
        name="locate_commits",
        description="定位候选提交范围，参数：目标版本号，是否允许等于"
    ),
    StructuredTool.from_function(
        func=cve_analyzer.get_next_diff,
        name="get_diff",
        description="获取下一个待分析的代码差异，返回（修订号, 差异内容）"
    )
]

# 初始化模型
client = OpenAI(
    base_url="https://xiaoai.plus/v1",
    api_key="sk-u3ph8f8eivL6YnttJdgwM8imj3lPE0MPJs5SrF1nTkqJZkXs",
    timeout=200
)

def analyze_diff_with_llm(diff: str, cve_desc: str) -> Tuple[str, str]:
    """增强的大模型分析"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"""请严格分析代码差异是否完全修复CVE漏洞：
                ### CVE描述
                {cve_desc}
                
                ### 代码差异
                {diff}
                
                ### 验证要求
                1. 必须完全覆盖CVE描述中的所有攻击路径
                2. 新增的安全检查必须与漏洞直接相关
                3. 部分修复或无关修改视为未修复
                
                ### 输出格式
                判断结果：修复/未修复
                分析依据：分点说明匹配情况"""
            }],
            temperature=0.3
        )
        content = response.choices[0].message.content
        repair_pattern = re.compile(r'判断结果\s*[:：]\s*(修复|已修复|fixed)', re.IGNORECASE)
        if repair_pattern.search(content):
           return (content, "修复")
    except Exception as e:
        logger.error(f"大模型分析失败：{str(e)}")
        return ("分析超时", "未修复")

# ================== Agent配置 ==================
SYSTEM_PROMPT = """作为安全分析专家，请按以下严谨流程工作：
1. 使用load_cve_data加载漏洞信息（必须验证CVE ID格式）
2. 用parse_version精确解析影响版本条件
3. 通过fetch_svn_logs获取完整提交历史
4. 用locate_commits确定候选提交范围（结合版本条件）
5. 循环使用get_diff获取差异（最多5次）
6. 严格分析差异是否符合CVE修复标准

验证规则：
- 必须验证版本匹配的严格性（如1.2 vs 1.2.0）
- 优先检查包含安全关键词的提交
- 必须确认修改覆盖所有漏洞点
- 遇到错误尝试下一个提交
"""

model = ChatOpenAI(
    model="gpt-4",
    base_url="https://xiaoai.plus/v1",
    api_key="sk-u3ph8f8eivL6YnttJdgwM8imj3lPE0MPJs5SrF1nTkqJZkXs",
    temperature=0.3
).bind(
    tools=[convert_to_openai_tool(tool) for tool in tools],
    tool_choice="auto"
)

agent = create_react_agent(model, tools)

# ================== 执行入口 ==================
if __name__ == "__main__":
    try:
        response = agent.invoke({
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="请分析CVE-2022-0313的修复提交")
            ]
        })
        
        final_result = "未找到修复提交"
        tool_call_mapping = {}

        # 解析大模型响应
        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    tool_call_mapping[call['id']] = call['name']
            
            if isinstance(msg, ToolMessage):
                tool_name = tool_call_mapping.get(msg.tool_call_id)
                if tool_name == "get_diff":
                    try:
                        revision, diff = eval(msg.content)  # 实际应使用更安全的解析方法
                        analysis, conclusion = analyze_diff_with_llm(diff, cve_analyzer.current_cve["description"])
                        if conclusion == "修复":
                            final_result = f"找到修复提交：{revision}"
                            logger.info(f"\n分析结果：{final_result}\n分析依据：{analysis}")
                            break
                    except Exception as e:
                        logger.error(f"内容解析失败：{str(e)}")

        print(final_result)

    except Exception as e:
        logger.error(f"分析流程失败：{str(e)}")
        print("分析过程中发生错误，请检查日志")
        
     