import os
import sys
import io
import json
import time
import queue
import logging
import threading
import secrets
from logging.handlers import RotatingFileHandler
import requests
from flask import Flask, request, Response, render_template, jsonify, session, redirect, url_for, render_template_string
from functools import wraps
from openai import OpenAI

# 本地测试时可取消下面这行注释
# from waitress import serve

# ================= 日志配置 =================
LOG_FILE = os.path.join(os.path.dirname(__file__), "server.log")
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# ================= 配置区域（全部从环境变量读取） =================
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_KIMI = "moonshotai/kimi-k3"

AGNES_CHAT_API_KEY = os.environ.get("AGNES_CHAT_API_KEY", "")
AGNES_BASE_URL = "https://api.agnes-ai.cn/v1"
MODEL_AGNES = "agnes-3.0-flash"

AGNES_SOLUTION_API_KEY = os.environ.get("AGNES_SOLUTION_API_KEY", "")
MODEL_SOLUTION = "agnes-3.0-flash"

client_agnes_chat = OpenAI(api_key=AGNES_CHAT_API_KEY, base_url=AGNES_BASE_URL, timeout=300)
client_agnes_solution = OpenAI(api_key=AGNES_SOLUTION_API_KEY, base_url=AGNES_BASE_URL, timeout=300)

app = Flask(__name__)
app.json.ensure_ascii = False

# ================= 登录配置 =================
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "admin")
LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "change_me")

# 安全 Cookie 配置
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=86400,
)

# ================= 系统提示词 =================
SYSTEM_PROMPT_CHAT = """你是一个编程助手。请根据当前对话上下文判断用户意图：

【语言强制要求（最高优先级）】
你必须在思考过程和最终回答中全程使用简体中文。
禁止使用英文进行推理。即使题目是英文，也要翻译成中文后再思考。
代码本身和变量名可以用英文，但解释和推理必须用中文。

【情况一：第一次解题】
如果用户给出了完整的题目描述，请按照以下要求输出：
1. 【思路解释】先给出解题思路，分析时间和空间复杂度。
2. 【代码实现】提供完整的代码。代码必须严格遵守以下格式规范：
   - 小括号两侧不加多余空格。正确示例：if(x > 0)、for(int i = 0; i < n; i++)。
   - 大括号前面不加空格。正确示例：if(x > 0){、void func(){。
   - 所有运算符两侧各加一个空格。正确示例：a + b、x == y、i < n。

【情况二：修改已有代码】
如果用户是在要求你修改、优化、调试已有的代码，则：
1. 不需要重新解释整体思路和复杂度。
2. 只需要说明你修改了哪些部分，并给出修改后的代码。
3. 代码的格式规范依然必须严格遵守（同上三条）。

无论哪种情况，在输出代码前，请默默检查一遍是否符合格式要求。"""

SYSTEM_PROMPT_SOLUTION = """你是一个专业的洛谷题解格式修正助手。你的任务是根据洛谷官方的学术规范、排版指南、LaTeX 格式手册和 Markdown 格式手册，对用户提供的题解草稿进行格式修正。

【最重要的核心原则（必须严格遵守）】
1. **只能修改格式，绝对不能修改任何内容相关的东西。**
2. 不得添加、删除、替换、改写任何文字、句子、段落、代码、公式的语义。
3. 不得添加原文中没有的解释、说明、过渡句、总结句。
4. 不得删除原文中的任何内容，包括看似冗余的表达。
5. 不得修改代码逻辑、变量名、注释文字、算法思路。
6. 不得修改数学公式中的变量名、系数、运算符号的语义。
7. 如果发现原文存在内容错误，不要擅自修改，只在文末用一行方括号注明 `[提示：原文可能存在内容问题，但按要求未改动]`。
8. 你的所有操作仅限于：标点符号、空格、换行、标题层级、LaTeX 定界符、Markdown 语法标记、代码块格式、列表格式。

【修正规则（只针对格式）】
一、基本标点与空格
1. 中文必须使用全角标点符号，英文使用半角标点符号。
2. 中文与英文、数字或公式之间以一个半角空格隔开。
3. 中文标点符号与英文、数字或公式之间不应有空格。
4. 每句话末尾必须添加句号。

二、标题与结构
1. 使用 #, ##, ###, #### 表示标题。
2. 建议只用一个一级标题作为题解标题，其余内容使用二级标题。
3. 标题的文字内容不得修改，只能调整标题的层级标记。

三、数学公式（LaTeX）
1. 数学公式必须使用 LaTeX，非数学内容不应使用 LaTeX。
2. 行内公式使用 $ $ 定界，行间公式使用 $$ $$ 定界且独立成行。
3. 赋值语句应使用 \\gets 或 \\to。
4. 整除使用 \\lfloor \\frac{a}{b} \\rfloor。
5. 取模使用 a \\bmod b 或 a \\equiv b \\pmod p。
6. 特定函数名使用正体，如 \\gcd, \\max, \\min, \\log, \\det。
7. 大数字使用科学计数法，如 5 \\times 10^9。
8. 公式内只能使用英文半角标点，公式外只能使用中文全角标点。

四、Markdown 排版
1. 使用 - 表示无序列表，使用 1. 表示有序列表。
2. 使用行内代码块表示字符串或代码。
3. 代码块使用三个反引号包裹，并指定语言。
4. 段落之间必须有空行。

【输出要求】
1. 直接输出修正后的完整题解（Markdown 格式），不要添加任何额外的解释或说明。
2. 确保修正后的内容与原文在语义上完全一致，只调整了格式。
3. 如果发现内容问题，在文末单独加一行：`[提示：...]`，简明说明未改动的内容问题。
4. 确保修正后的内容可以直接复制到洛谷平台使用。"""

# ================= 多会话存储 =================
chat_sessions = {}
solution_sessions = {}
session_lock = threading.Lock()

def get_chat_session(session_id):
    with session_lock:
        if session_id not in chat_sessions:
            chat_sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
        return chat_sessions[session_id]

def get_solution_session(session_id):
    with session_lock:
        if session_id not in solution_sessions:
            solution_sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT_SOLUTION}]
        return solution_sessions[session_id]

# ================= Kimi K3 流式 =================
def stream_kimi(messages, temperature, max_tokens, reasoning_effort):
    payload = {
        "messages": messages,
        "model": MODEL_KIMI,
        "max_tokens": max_tokens,
        "seed": 0,
        "stream": True,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    headers = {
        "Authorization": "Bearer " + NVIDIA_API_KEY,
        "Accept": "text/event-stream",
        "Content-Type": "application/json"
    }

    q = queue.Queue()

    def fetch():
        try:
            logging.info(f"[Kimi K3] 发起请求（reasoning_effort={reasoning_effort}）...")
            resp = requests.post(
                NVIDIA_INVOKE_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(30, 900)
            )
            logging.info(f"[Kimi K3] 状态码: {resp.status_code}")
            if resp.status_code != 200:
                q.put(("error", f"HTTP {resp.status_code}: {resp.text[:500]}"))
                return
            for line in resp.iter_lines():
                q.put(("line", line))
            q.put(("done", None))
        except requests.exceptions.Timeout:
            q.put(("error", "请求超时"))
        except requests.exceptions.ConnectionError as e:
            q.put(("error", f"无法连接到 NVIDIA API: {e}"))
        except Exception as e:
            q.put(("error", f"请求异常: {e}"))

    thread = threading.Thread(target=fetch, daemon=True)
    thread.start()

    full_reply = ""
    while True:
        try:
            msg_type, msg_data = q.get(timeout=10)
        except queue.Empty:
            yield ": keep-alive\n\n"
            continue

        if msg_type == "error":
            yield f"data: {json.dumps({'error': msg_data}, ensure_ascii=False)}\n\n"
            return
        elif msg_type == "done":
            break
        elif msg_type == "line":
            line = msg_data
            if not line:
                continue
            try:
                line_str = line.decode("utf-8")
            except Exception:
                continue
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    yield f"data: {json.dumps({'reasoning': reasoning}, ensure_ascii=False)}\n\n"
                content_piece = delta.get("content", "")
                if content_piece:
                    full_reply += content_piece
                    yield f"data: {json.dumps({'content': content_piece}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logging.warning(f"解析 chunk 失败: {e}")
                continue

    messages.append({"role": "assistant", "content": full_reply})
    yield "data: [DONE]\n\n"

# ================= Agnes 流式 =================
def stream_agnes_chat(messages, temperature, max_tokens):
    full_reply = ""
    try:
        response = client_agnes_chat.chat.completions.create(
            model=MODEL_AGNES,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_reply += delta
                yield f"data: {json.dumps({'content': delta}, ensure_ascii=False)}\n\n"
        messages.append({"role": "assistant", "content": full_reply})
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

# ================= 登录路由 =================
LOGIN_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>登录 - AI 助手</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .login-box {
            background: #fff;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.2);
            width: 360px;
            max-width: 90%;
        }
        .login-box h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
            font-size: 24px;
        }
        .login-box input {
            width: 100%;
            padding: 12px 16px;
            margin-bottom: 16px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 15px;
            outline: none;
            transition: border-color 0.2s;
        }
        .login-box input:focus { border-color: #667eea; }
        .login-box button {
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .login-box button:hover { background: #5568d3; }
        .error { color: red; text-align: center; margin-bottom: 16px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>🔐 请登录</h1>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="post">
            <input type="text" name="username" placeholder="用户名" required>
            <input type="password" name="password" placeholder="密码" required>
            <button type="submit">登录</button>
        </form>
    </div>
</body>
</html>
"""

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            session["logged_in"] = True
            session.permanent = True
            return redirect(url_for("index"))
        else:
            return render_template_string(LOGIN_PAGE, error="用户名或密码错误")
    return render_template_string(LOGIN_PAGE)

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ================= 主路由 =================

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    user_input = data.get("message", "").strip()
    images = data.get("images", [])
    model_choice = data.get("model", "agnes")
    reasoning_effort = data.get("reasoning_effort", "low")
    if reasoning_effort not in ("low", "high", "max"):
        reasoning_effort = "low"

    if not user_input and not images:
        return jsonify({"error": "empty"}), 400

    temperature = float(data.get("temperature", 0.2))
    max_tokens = int(data.get("max_tokens", 16384))
    max_tokens = max(256, min(max_tokens, 65536))

    messages = get_chat_session(session_id)

    content = []
    if user_input:
        content.append({"type": "text", "text": user_input})
    for img_base64 in images:
        content.append({"type": "image_url", "image_url": {"url": img_base64}})

    if len(content) == 1 and content[0]["type"] == "text":
        messages.append({"role": "user", "content": user_input})
    else:
        messages.append({"role": "user", "content": content})

    if model_choice == "kimi":
        return Response(
            stream_kimi(messages, temperature, max_tokens, reasoning_effort),
            mimetype="text/event-stream"
        )
    else:
        return Response(
            stream_agnes_chat(messages, temperature, max_tokens),
            mimetype="text/event-stream"
        )

@app.route("/clear", methods=["POST"])
@login_required
def clear():
    data = request.json or {}
    session_id = data.get("session_id", "default")
    with session_lock:
        chat_sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    return jsonify({"status": "ok"})

@app.route("/solution", methods=["POST"])
@login_required
def solution():
    data = request.json
    session_id = data.get("session_id", "default")
    user_content = data.get("content", "").strip()
    if not user_content:
        return jsonify({"error": "empty"}), 400

    messages = get_solution_session(session_id)
    messages.append({"role": "user", "content": user_content})

    def generate():
        full_reply = ""
        try:
            response = client_agnes_solution.chat.completions.create(
                model=MODEL_SOLUTION,
                messages=messages,
                temperature=0.1,
                max_tokens=16384,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_reply += delta
                    yield f"data: {json.dumps({'content': delta}, ensure_ascii=False)}\n\n"
            messages.append({"role": "assistant", "content": full_reply})
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/clear_solution", methods=["POST"])
@login_required
def clear_solution():
    data = request.json or {}
    session_id = data.get("session_id", "default")
    with session_lock:
        solution_sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT_SOLUTION}]
    return jsonify({"status": "ok"})

# ================= 启动（本地测试） =================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 1145))   # Render 会提供 PORT 环境变量，本地测试时默认 1145
    print("=" * 50)
    print(f"AI 助手已启动，监听端口: {port}")
    print("=" * 50)
    # 本地测试用 waitress，云端部署时 app.run 会被 Render 接管
    # serve(app, host="0.0.0.0", port=port, threads=16, connection_limit=200, channel_timeout=900)
    app.run(host="0.0.0.0", port=port, debug=False)
