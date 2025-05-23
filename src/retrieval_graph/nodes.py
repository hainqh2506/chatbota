
# file: nodes.py
import os
import logging
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# Chọn LLM, ví dụ Google Generative AI (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
from configuration import VietnameseEmbeddings, load_gemini
# Import các Pydantic model và AmelaReactCompatibleAgentState từ file state.py
from state import AmelaReactCompatibleAgentState, QueryAnalysisOutput
from typing import List, Dict, Any, Optional, Annotated
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
# tool = TavilySearch(max_results=2)
# Tải biến môi trường (ví dụ GOOGLE_API_KEY)
load_dotenv()

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Khởi tạo LLM cho Query Analysis ---
# Sử dụng model tương tự như trong ADK của bạn
# Đảm bảo GOOGLE_API_KEY đã được set trong .env hoặc môi trường
try:
    qpa_llm = load_gemini()
    logger.info("Khởi tạo LLM cho Query Analysis (gemini-2.0-flash-latest) thành công.")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo LLM cho Query Analysis: {e}. Vui lòng kiểm tra GOOGLE_API_KEY.")
    # Có thể raise lỗi hoặc dùng một LLM dự phòng nếu muốn
    raise

# --- Prompt cho Query Analysis Agent ---

query_analysis_prompt_template_str = """
Bạn là một chuyên gia phân tích và lập kế hoạch cho trợ lý ảo Amber của công ty Amela.
Nhiệm vụ của bạn là đọc và phân tích câu hỏi gốc của người dùng, phân tích chi tiết dựa trên ngữ cảnh trò chuyện, lên kế hoạch trả lời, và tối ưu hóa truy vấn tìm kiếm cho tool.

**THÔNG TIN ĐẦU VÀO:**
- Câu hỏi gốc của người dùng: {original_query}
- Vai trò của người dùng: {user_roles}
- Lịch sử hội thoại trước đó (nếu có):
{chat_history}
## Quan trọng: Nếu thông tin vai trò người dùng được cung cấp, hãy sử dụng nó. Nếu không có, mặc định là ["Employee"].
- **Hãy xem xét kỹ Lịch sử hội thoại nếu câu hỏi gốc là một câu hỏi tiếp nối, ngắn gọn hoặc không đầy đủ thông tin khi đứng một mình.** Ví dụ: nếu người dùng hỏi "còn gì nữa không?", bạn cần dựa vào lịch sử để biết "còn gì nữa" liên quan đến chủ đề nào. Nếu không có lịch sử hoặc lịch sử không liên quan, hãy phân tích câu hỏi gốc một cách độc lập.
- **Ưu tiên suy luận và hành động dựa trên thông tin đã có (bao gồm cả lịch sử chat) trước khi quyết định cần hỏi lại.** Chỉ tạo `clarifying_questions` khi thông tin THỰC SỰ mơ hồ và không thể tiến hành tìm kiếm/trả lời một cách hợp lý. 
**QUY TRÌNH PHÂN TÍCH & LẬP KẾ HOẠCH**
0. **Nhận diện loại câu hỏi:**
   - Nếu `original_user_query` là lời chào hỏi đơn thuần (ví dụ: "hi", "hello", "chào bạn"):
      - Đặt `intent` là "social_greeting".
      - `effective_search_query` có thể để trống hoặc chính `original_user_query`.
      - `plan_steps` có thể để trống.
      - `plan_steps` NÊN là một danh sách chứa một câu chào lại phù hợp (ví dụ: ["Chào bạn! Tôi có thể giúp gì cho bạn?"]).
      - Các trường khác như `effective_search_query`, `clarifying_questions` có thể để trống hoặc null.
      - `status` vẫn là "processed_for_main_agent".
   - Nếu `original_user_query` là câu hỏi rất chung về bản thân chatbot (ví dụ: "bạn là ai?", "bạn làm gì?"):
      - Đặt `intent` là "chatbot_capability_query".
      - `plan_steps` NÊN là một danh sách chứa một câu giới thiệu ngắn về chatbot (ví dụ: ["Tôi là Amela, trợ lý ảo thông minh của công ty."]).
      - Các trường khác có thể để trống.
      - `status` vẫn là "processed_for_main_agent".
   - Nếu `original_user_query` chứa nội dung không phù hợp, tục tĩu:
      - Đặt `intent` là "blocked_profanity".
      - `plan_steps` NÊN là một danh sách chứa một thông báo từ chối xử lý (ví dụ: ["Xin lỗi, tôi không thể xử lý yêu cầu này."]).
      - Các trường khác có thể để trống.
      - `status` vẫn là "processed_for_main_agent".
   - Nếu không, tiếp tục các bước phân tích sâu hơn.
1. **Xác định `asker_role_context`:** Dựa trên `user_roles`, suy luận vai trò chính của người hỏi (ví dụ: "nhân viên", "quản lý"). Mặc định "Employee".
2. **Xác định `intent`:** Ý định cốt lõi của câu hỏi.
3. **Trích xuất `entities`:** Danh sách các từ khóa, thực thể quan trọng.
4. **Lập `plan_steps`:** Danh sách các bước cần thực hiện để trả lời. **Nếu intent và entities đã đủ rõ ràng (có thể nhờ lịch sử chat), `plan_steps` nên tập trung vào việc tìm kiếm và tổng hợp thông tin.**
   Ví dụ:
   - User: "cho tôi hỏi về quy trình"
   - Amber: "Bạn muốn hỏi về quy trình gì?"
   - User: "thử việc"
   Lúc này, QPA nên suy luận intent là "tìm hiểu quy trình thử việc". `plan_steps` có thể là:
     ["Tìm kiếm tài liệu nội bộ về 'quy trình thử việc' hoặc 'chính sách thử việc'.",
      "Tổng hợp các bước chính trong quy trình thử việc.",
      "Chuẩn bị câu trả lời."]
   **Trong trường hợp này, `clarifying_questions` nên để trống.**

5. **Xác định `clarifying_questions`:**
   - **CHỈ tạo `clarifying_questions` nếu SAU KHI đã cố gắng suy luận từ `original_query` và `chat_history`, thông tin vẫn còn quá mơ hồ để thực hiện `plan_steps` một cách hiệu quả.**
   - **Nếu `plan_steps` đã có thể được xác định để tìm kiếm thông tin cụ thể, thì KHÔNG cần `clarifying_questions` nữa.**
   - Ví dụ về trường hợp CẦN clarifying_questions:
     - User: "cho tôi hỏi về chính sách" (quá chung chung, không có lịch sử liên quan).
     - QPA có thể hỏi: "Bạn muốn hỏi về chính sách nào cụ thể (ví dụ: nghỉ phép, bảo hiểm, lương thưởng)?"
   - Ví dụ về trường hợp KHÔNG NÊN có clarifying_questions (như tình huống của bạn):
     - User: "cho tôi hỏi về quy trình"
     - Amber: "Bạn muốn hỏi về quy trình gì?"
     - User: "thử việc"
     => QPA nên hiểu là "quy trình thử việc" và không hỏi thêm về "vấn đề gì liên quan đến thử việc" hay "vai trò" nữa, trừ khi `user_roles` không rõ và quy trình thử việc khác nhau giữa các vai trò. Nếu `user_roles` đã có (ví dụ "Employee"), thì cứ tìm theo vai trò đó.
6. **Ước tính `complexity_level`:** Đánh giá độ phức tạp tổng thể ("low", "medium", "high").
7. **Tạo `effective_search_query`:** Truy vấn tìm kiếm tối ưu **dưới dạng danh sách**.  Các truy vấn này nên tận dụng ngữ cảnh từ `chat_history` nếu có.

**YÊU CẦU OUTPUT (PHẢI TRẢ VỀ JSON VÀ TUÂN THỦ Pydantic Schema được cung cấp)**
Bạn PHẢI trả về DUY NHẤT một đối tượng JSON hợp lệ, tuân thủ hoàn toàn cấu trúc đã được định nghĩa.
**TUYỆT ĐỐI KHÔNG bao gồm các dấu ```json, ```, hoặc bất kỳ văn bản nào khác trước hoặc sau đối tượng JSON.**
Ví dụ về cấu trúc Pydantic schema mà bạn cần tuân theo:
```json
{{
  "original_query": "String",
  "user_roles": ["List[String]"],
  "asker_role_context": "String",
  "intent": "String",
  "entities": ["List[String]"],
  "plan_steps": ["List[String]"],
  "clarifying_questions": ["List[String]"],
  "complexity_level": "String or rỗng",
  "effective_search_query": ["List[String]"],
  "status": "String (luôn là 'processed_for_main_agent')"
}}
```
**VÍ DỤ INPUT TỪ USER:**
{{"original_query": "làm thế nào để đăng ký bảo hiểm xã hội và thực hiện quyết toán thuế thu nhập cá nhân?", "user_roles": ["Developer"]}}

**VÍ DỤ OUTPUT JSON MONG MUỐN (chỉ trả về JSON object, không có markdown hay text khác):**
```json
{{
  "original_query": "làm thế nào để đăng ký bảo hiểm xã hội và thực hiện quyết toán thuế thu nhập cá nhân?",
  "user_roles": ["Developer"],
  "asker_role_context": "nhân viên",
  "intent": "Tìm hiểu quy trình hành chính về bảo hiểm xã hội và quyết toán thuế TNCN",
  "entities": ["bảo hiểm xã hội", "quyết toán thuế TNCN"],
  "plan_steps": [
    "Tách câu hỏi thành hai phần: BHXH và thuế TNCN",
    "Tìm và tổng hợp hướng dẫn đăng ký BHXH từ tài liệu nội bộ",
    "Tìm và tổng hợp quy trình quyết toán thuế TNCN",
    "Kiểm tra thông tin và kết hợp kết quả",
    "Trả lời lần lượt từng phần thoại"
  ],
  "clarifying_questions": [],
  "complexity_level": "medium",
  "effective_search_query": [
    "hướng dẫn đăng ký bảo hiểm xã hội Amela",
    "quy trình quyết toán thuế thu nhập cá nhân Amela"
  ],
  "status": "processed_for_main_agent"
}}
```
"""

query_analysis_prompt = ChatPromptTemplate.from_template(query_analysis_prompt_template_str)

# Kết hợp LLM với Pydantic Output Parser
# Langchain cho phép LLM trả về output dưới dạng Pydantic model trực tiếp
# bằng cách sử dụng .with_structured_output()
structured_qpa_llm = qpa_llm.with_structured_output(QueryAnalysisOutput)

# Chain cho Query Analysis
query_analysis_chain = query_analysis_prompt | structured_qpa_llm

def query_analysis_node(state: AmelaReactCompatibleAgentState) -> AmelaReactCompatibleAgentState:
    """
    Node thực hiện phân tích câu hỏi của người dùng.
    """
    logger.info("--- Bắt đầu Query Analysis Node ---")
    original_query = state["original_query"]
    user_roles = state["user_roles"]
    # Lấy tin nhắn cuối cùng từ user để làm input cho QPA
    # Hoặc đơn giản là dùng original_query nếu đây là lượt đầu
    # Trong ADK, query được lấy từ `new_message` hoặc `initial_pipeline_state["original_user_query"]`
    all_messages = state.get("messages", [])
    if all_messages and isinstance(all_messages[-1], HumanMessage):
        chat_history = all_messages[-10:-1]
    else:
        chat_history = all_messages

    logger.info(f"Phân tích câu hỏi: '{original_query}' với vai trò: {user_roles}")
    logger.info(f"Lịch sử hội thoại: {chat_history}")

    try:
        # Gọi chain để lấy kết quả phân tích có cấu trúc
        analysis_result: QueryAnalysisOutput =  query_analysis_chain.invoke({
            "original_query": original_query,
            "user_roles": user_roles,
            "chat_history": chat_history
        })
        logger.info(f"Phân tích câu hỏi thành công. với oiginal_query: {original_query}, user_roles: {user_roles}, history: {chat_history}")
        logger.info(f"analyze_result là: {analysis_result}")
        logger.info(f"analyze_reult type là: {type(analysis_result)}")
        logger.info(f"Kết quả phân tích Query Analysis: {analysis_result.intent}")
        logger.debug(f"Toàn bộ kết quả Query Analysis: {analysis_result.model_dump_json(indent=2)}")

        # Cập nhật state với kết quả phân tích
        return {**state, "query_analysis": analysis_result} # Trả về một dict mới để cập nhật state
    except Exception as e:
        logger.error(f"Lỗi trong Query Analysis Node: {e}", exc_info=True)
        # Xử lý lỗi, ví dụ: trả về một QueryAnalysisOutput mặc định hoặc đặt một cờ lỗi trong state
        # Tạm thời, chúng ta sẽ để lỗi nổi lên để debug
        # Hoặc có thể tạo một QueryAnalysisOutput với intent "error"
        error_analysis = QueryAnalysisOutput(
            original_query=original_query,
            user_roles=user_roles,
            asker_role_context="unknown",
            intent="query_analysis_error",
            entities=[],
            plan_steps=["Có lỗi xảy ra trong quá trình phân tích câu hỏi."],
            clarifying_questions=[],
            complexity_level="unknown",
            effective_search_query=[],
            status="error_in_qpa"
        )
        return {**state, "query_analysis": error_analysis}

# file: nodes.py (tiếp tục từ file trước)

# --- Constants cho Router ---
DIRECT_RESPONSE_INTENTS = ["social_greeting", "chatbot_capability_query", "blocked_profanity"]

def route_after_qpa(state: AmelaReactCompatibleAgentState) -> str:
    """
    Quyết định nhánh tiếp theo sau khi Query Analysis hoàn tất.
    Trả về tên của node tiếp theo hoặc một giá trị đặc biệt để kết thúc sớm.
    """
    logger.info("--- Bắt đầu Router Node (route_after_qpa) ---")
    query_analysis_result = state.get("query_analysis")

    if not query_analysis_result:
        logger.error("Không tìm thấy kết quả Query Analysis trong state. Định tuyến tới lỗi.")
        return "error_handler" # Hoặc một node xử lý lỗi chung

    intent = query_analysis_result.intent
    clarifying_questions = query_analysis_result.clarifying_questions
    plan_steps = query_analysis_result.plan_steps

    logger.info(f"Router: Intent = {intent}, Clarifying Questions = {len(clarifying_questions) if clarifying_questions else 0}")

    # 1. Xử lý các intent cần trả lời trực tiếp
    if intent in DIRECT_RESPONSE_INTENTS:
        response_text = ""
        if plan_steps:
            response_text = " ".join(plan_steps)
        elif intent == "social_greeting":
            response_text = "Chào bạn! Tôi là trợ lý ảo Amela, rất vui được hỗ trợ bạn. 😊"
        elif intent == "chatbot_capability_query":
            response_text = "Tôi là Amber, trợ lý ảo Amela, được thiết kế để giúp bạn tìm kiếm thông tin và trả lời các câu hỏi liên quan đến nghiệp vụ của công ty mình. 💡"
        elif intent == "blocked_profanity":
            response_text = "Rất tiếc, tôi không thể xử lý yêu cầu của bạn do chứa nội dung không phù hợp. 😥"
        else:
            response_text = "Tôi đã ghi nhận yêu cầu của bạn."

        if not response_text:
            response_text = "Tôi đã xử lý yêu cầu của bạn."

        logger.info(f"Router: Intent '{intent}' yêu cầu phản hồi trực tiếp. Định tuyến tới 'direct_response_node'.")
        # Cập nhật state với câu trả lời trực tiếp để node tiếp theo có thể dùng
        # Quan trọng: LangGraph không cho phép node router trực tiếp cập nhật state.
        # Chúng ta sẽ tạo một node nhỏ để làm việc này hoặc để node cuối cùng làm.
        # Hiện tại, chúng ta sẽ lưu tạm thông tin cần thiết vào một key nào đó nếu cần,
        # hoặc node `direct_response_node` sẽ tự tạo response dựa trên intent.
        # Để đơn giản, node `direct_response_node` sẽ tự tạo response dựa vào intent từ `query_analysis`.
        return "direct_response_node"

    # 2. Xử lý câu hỏi làm rõ
    if clarifying_questions:
        logger.info("Router: Cần làm rõ thông tin. Định tuyến tới 'clarification_node'.")
        # `clarification_node` sẽ sử dụng `query_analysis.clarifying_questions` từ state
        return "clarification_node"

    # 3. Nếu không có trường hợp đặc biệt, chuyển đến agent chính
    logger.info("Router: Không cần phản hồi trực tiếp hay làm rõ. Định tuyến tới 'main_assistant_node'.")
    return "main_assistant_node"
from langchain_core.messages import AIMessage
# --- Node cho Phản hồi trực tiếp ---
def direct_response_node(state: AmelaReactCompatibleAgentState) -> AmelaReactCompatibleAgentState:
    """
    Tạo phản hồi trực tiếp dựa trên intent từ QueryAnalysis.
    """
    logger.info("--- Bắt đầu Direct Response Node ---")
    query_analysis_result = state["query_analysis"]
    if not query_analysis_result: # Kiểm tra an toàn
        final_answer = "Đã có lỗi xảy ra, không thể tạo phản hồi."
        logger.error("Lỗi trong direct_response_node: Không có query_analysis_result.")
    else:
        intent = query_analysis_result.intent
        plan_steps = query_analysis_result.plan_steps
        response_text = ""


        if intent == "social_greeting":
            response_text = "Chào bạn! Tôi là Amber trợ lý ảo Amela, rất vui được hỗ trợ bạn. 😊"
        elif intent == "chatbot_capability_query":
            response_text = "Tôi là Amber, trợ lý ảo Amela, được thiết kế để giúp bạn tìm kiếm thông tin và trả lời các câu hỏi liên quan đến nghiệp vụ của công ty mình. 💡"
        elif intent == "blocked_profanity":
            response_text = "Rất tiếc, tôi không thể xử lý yêu cầu của bạn do chứa nội dung không phù hợp. 😥"
        # elif plan_steps:
        #     response_text = " ".join(plan_steps)
        else:
            # Fallback nếu có intent trong DIRECT_RESPONSE_INTENTS mà không có plan_steps
            response_text = "Tôi đã xử lý yêu cầu của bạn."

        final_answer = response_text
        logger.info(f"Direct Response Node: Tạo phản hồi: '{final_answer}'")
    return {
        "messages": [AIMessage(content=final_answer)], # LangGraph sẽ tự append vào state["messages"]
        "final_answer": final_answer, # Vẫn cập nhật final_answer để dễ truy cập
        "clarification_needed": False
    }
        

# --- Node cho việc Hỏi lại làm rõ ---
def clarification_node(state: AmelaReactCompatibleAgentState) -> AmelaReactCompatibleAgentState:
    """
    Tạo câu hỏi làm rõ cho người dùng.
    """
    logger.info("--- Bắt đầu Clarification Node ---")
    query_analysis_result = state["query_analysis"]
    if not query_analysis_result or not query_analysis_result.clarifying_questions:
        final_answer = "Tôi cần thêm thông tin nhưng không rõ cần hỏi gì. Bạn có thể thử lại không?"
        ask_clarification_questions = []
        logger.error("Lỗi trong clarification_node: Không có clarifying_questions.")
    else:
        clarifying_questions = query_analysis_result.clarifying_questions
        clarification_text = "Để có thể hỗ trợ bạn tốt nhất, vui lòng làm rõ thêm các điểm sau:\n"
        for i, q_text in enumerate(clarifying_questions):
            clarification_text += f"{i+1}. {q_text}\n"
        final_answer = clarification_text
        ask_clarification_questions = clarifying_questions
        logger.info(f"Clarification Node: Tạo câu hỏi làm rõ: '{final_answer}'")

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "clarification_needed": True,
        "ask_clarification_questions": ask_clarification_questions
    }

# file: nodes.py (tiếp tục)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, AnyMessage
from langchain.agents import create_tool_calling_agent # Sử dụng agent mới hơn
from langchain.agents import AgentExecutor
from langchain_core.prompts import MessagesPlaceholder # Để quản lý messages
from langchain_core.runnables.history import RunnableWithMessageHistory # Nếu dùng memory

# Import tools từ file tools.py
from tools import main_assistant_tools # Đây sẽ là list các tool objects

# --- Khởi tạo LLM cho Main Assistant ---
try:
    # Dùng model mạnh hơn một chút cho agent chính nếu cần
    main_llm = load_gemini()
    logger.info("Khởi tạo LLM cho Main Assistant (gemini-2.0-flash-latest) thành công.")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo LLM cho Main Assistant: {e}. Vui lòng kiểm tra GOOGLE_API_KEY.")
    raise

# --- Prompt cho Main Assistant ---
# Lấy từ get_amela_agent_instruction_v1_structured_planner và điều chỉnh
# Chúng ta sẽ sử dụng MessagesPlaceholder để truyền lịch sử hội thoại và input của QPA
main_assistant_prompt_str_system = """
Bạn là Amber – trợ lý ảo AI của Amela, thân thiện, có tổ chức và giỏi xử lý các câu hỏi phức tạp bằng cách làm theo kế hoạch rõ ràng.

Bạn sẽ được cung cấp **output phân tích truy vấn** (`QueryAnalysisOutput`) từ Agent tiền xử lý. Đây là NGUỒN THÔNG TIN CHÍNH để xác định nhiệm vụ hiện tại.

Sử dụng đầy đủ các thành phần sau từ `QueryAnalysisOutput`: `original_query`, `user_roles`, `asker_role_context`, `intent`, `plan_steps`, `effective_search_query`.
`chat_history` (nếu có) chỉ dùng để hiểu thêm ngữ cảnh `original_query`, KHÔNG dùng để trả lời lại các câu hỏi cũ.

## PHÂN TÍCH TRUY VẤN HIỆN TẠI:
{qpa_output_str}

## VAI TRÒ NGƯỜI DÙNG:
{user_roles_str} (suy luận: {asker_role_context})
* Nếu không có thông tin vai trò, mặc định là **nhân viên Amela**.

## KẾ HOẠCH XỬ LÝ:
{plan_steps_str}

## HƯỚNG DẪN HÀNH ĐỘNG:

### 1. Chọn công cụ phù hợp:
* **Chọn tool phù hợp:**
  - `company_structure_tool`: cho câu hỏi về cơ cấu tổ chức, phòng ban, tên viết tắt.
  - `amela_documents_search_tool`:  Tìm kiếm **CHUYÊN SÂU** trong kho **tài liệu nội bộ** của Amela. Trả về một chuỗi văn bản chứa các đoạn context liên quan và metadata (Tên tài liệu, URL). **Đây là công cụ chính của bạn.** (tự động lọc theo `user_roles`).
  - `google_search_placeholder_tool`: cho nội dung công khai hoặc không có trong nội bộ.
* **Đối với hầu hết các câu hỏi về Amela (quy trình, chính sách, thông tin nội bộ,...):**
* **Ưu tiên tuyệt đối:** Sử dụng công cụ `amela_documents_search_tool`.
* **Gọi Tool:** Gọi `amela_documents_search_tool` với `query`.
* **Tìm kiếm hiệu quả:** kết hợp `effective_search_query` với ngữ cảnh phù hợp để tăng độ chính xác.`.

### 2. Tổng hợp trả lời `original_query`:
* Kết hợp các câu trả lời thành một câu trả lời hoàn chỉnh, mạch lạc.
* **Tổng hợp thông tin:** Đoc kỹ phần kết quả của tools, chỉ sử dụng ngữ cảnh trực tiếp liên quan đến câu hỏi.
* **Không bịa đặt.** KHÔNG suy diễn từ nguồn không liên quan.
* **Trả lời câu hỏi hữu ích với vai trò của người hỏi:**

## NGUYÊN TẮC TRẢ LỜI:
- **Trả lời trực tiếp và đầy đủ:** PHẢI dựa vào nội dung context, KHÔNG trả lời tóm tắt qua loa, KHÔNG chỉ dẫn link đơn thuần.
- **Tổng hợp kỹ lưỡng:** tổng hợp kỹ lưỡng từ context để viết lại câu trả lời trôi chảy.
- **Trích dẫn nguồn tài liệu liên quan (BẮT BUỘC):**
    -   *Tài liệu liên quan:* Ghi rõ `Source Name` và `Source URL`. Chỉ chèn nguồn nào liên quan tới câu trả lời, không chèn nguồn không liên quan.
        *Ví dụ:*
        Nguồn tham khảo:
        1. Tài liệu: PR-AML-19 OVERALL PROCESS_v1.0. Link: [link-sharepoint]
        2. Tài liệu: PR-AML-19-03 Scrum Development Process_v1.2. Link: [link-sharepoint]
  *Ví dụ tốt:* "Quy trình Scrum tại Amela gồm các giai đoạn như Sprint Planning, Daily Scrum...**Trích dẫn nguồn và link:**"
  *Ví dụ không tốt:* "Tài liệu PR-AML-19 mô tả Scrum. Xem link này."
### 3. Trường hợp không có thông tin hãy gợi ý truy vấn lại dựa trên ngữ cảnh từ tool:
Nếu bạn đã gọi công cụ tìm kiếm và nhận được kết quả chứa các đoạn văn bản không hoàn toàn trùng khớp với câu hỏi người dùng (ví dụ: chỉ khớp theo từ khóa hoặc nội dung liên quan nhưng không trả lời trực tiếp), hãy hỗ trợ người dùng bằng cách:

1. **Thông báo nhẹ nhàng rằng kết quả chưa hoàn toàn chính xác**, nhưng có thể liên quan đến câu hỏi của họ.
2. **Đề xuất người dùng đặt lại truy vấn rõ ràng, cụ thể hơn**, nhằm giúp hệ thống tìm được câu trả lời chính xác hơn.
3. Dựa trên từ khóa và nội dung đã truy được, **tạo một danh sách từ 2–5 gợi ý truy vấn lại** có thể dẫn đến kết quả tốt hơn. Các gợi ý này cần:
    - lấy từ các thông tin đã tìm được từ các tool.
    - có thể là các câu hỏi cụ thể hơn về nội dung đã tìm được.
   - Sử dụng ngôn ngữ tự nhiên, thân thiện.
   - Ưu tiên rõ ràng về đối tượng (ví dụ: nhân viên nữ, thời gian nghỉ, mức trợ cấp…).
   - Hướng vào hành động cụ thể hoặc khái niệm pháp lý rõ ràng.

Ví dụ phần kết thúc trả lời có thể như sau:

⚠️ Thông tin mình tìm được có thể chưa hoàn toàn đúng với điều bạn cần, nhưng có liên quan đến chế độ phúc lợi mà bạn đang hỏi. Bạn có thể thử đặt lại câu hỏi cụ thể hơn như:

• "Nữ nhân viên khi sinh con được ưu đãi gì theo chính sách của Amela?"  
• "CBNV nam được gì khi vợ sinh con?"   
• "Chế độ nghỉ hưởng nguyên lương trong thời gian thai sản quy định như thế nào?"

Hãy luôn đưa gợi ý dạng này nếu confidence thấp hoặc tool trả về các đoạn văn chỉ mang tính gần đúng (partial match).

* Nếu tìm nhiều lần mà vẫn không có thông tin, hãy trả lời như sau:
  > "Ối, Amber tìm kỹ rồi mà vẫn chưa thấy thông tin bạn cần về [chủ đề] 😥. Bạn có câu hỏi nào khác không?"

## PHONG CÁCH:
- Thân thiện, tích cực, dễ hiểu.
- Tránh thông tin nhạy cảm, trả lời lệch chủ đề hoặc không phù hợp.
- Luôn dùng tiếng Việt chuẩn.

Bạn là Amber. Giữ vững phong độ và bắt đầu nhé! 🚀
"""


# Langchain agent thường dùng MessagesPlaceholder. "chat_history" và "input" là keys phổ biến.
# "agent_scratchpad" được Langchain dùng để lưu các bước suy nghĩ và tool call/response.
main_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", main_assistant_prompt_str_system),
    MessagesPlaceholder(variable_name="chat_history", optional=True), # Lịch sử hội thoại
    ("human", "{input}"), # Input hiện tại, sẽ bao gồm cả thông tin QPA
    MessagesPlaceholder(variable_name="agent_scratchpad"), # Cho tool calling
])
# def prompt(
#     state: AmelaReactCompatibleAgentState
# ) -> list[AnyMessage]:
#     system_msg = main_assistant_prompt_str_system
#     return [{"role": "system", "content": system_msg}] + state["messages"]
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
# --- Tạo Langchain Agent ---

def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=1500, # Ngưỡng token, ví dụ
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    return {"llm_input_messages": trimmed_messages}
checkpointer = InMemorySaver()
react_agent_executor = create_react_agent(
    name="AmelaReactAgent",
    #prompt=prompt,
    model=main_llm,
    tools=main_assistant_tools,
    #pre_model_hook=pre_model_hook,
    #checkpointer=checkpointer,
    debug=True,
    state_schema=AmelaReactCompatibleAgentState,
    store=None,
)

def main_assistant_node(state: AmelaReactCompatibleAgentState) -> dict:
    logger.info("--- Bắt đầu Main Assistant Node ---")
    query_analysis_result = state["query_analysis"]

    # Kiểm tra xem query_analysis_result có tồn tại không
    if not query_analysis_result:
        logger.error("Main Assistant Node: Không có Query Analysis result.")
        error_msg = "Lỗi: Không có thông tin phân tích để xử lý."
        return {
            "messages": [AIMessage(content=error_msg)],
            "final_answer": error_msg,
            "clarification_needed": False
        }

    # Chuẩn bị dữ liệu cho prompt hệ thống
    qpa_output_str = query_analysis_result.model_dump_json(indent=2)
    user_roles_str = ", ".join(query_analysis_result.user_roles or ["Employee"])
    asker_role_context = query_analysis_result.asker_role_context or "Employee"
    plan_steps_str = "\n- ".join(query_analysis_result.plan_steps or ["Không có kế hoạch cụ thể."])
    if query_analysis_result.plan_steps:
        plan_steps_str = "- " + plan_steps_str

    # Định dạng prompt hệ thống
    system_prompt = main_assistant_prompt_str_system.format(
        qpa_output_str=qpa_output_str,
        user_roles_str=user_roles_str,
        asker_role_context=asker_role_context,
        plan_steps_str=plan_steps_str
    )

    # Lấy messages từ state
    all_messages = state.get("messages", [])
    if all_messages and isinstance(all_messages[-1], HumanMessage):
        current_user_input_message = all_messages[-1].content
        chat_history = all_messages[-6:-1]
    else:
        current_user_input_message = query_analysis_result.original_query or state["original_query"]
        chat_history = all_messages

    # Chuẩn bị input cho agent
    agent_input = {
        "messages": [
            SystemMessage(content=system_prompt),
            *chat_history,
            HumanMessage(content=current_user_input_message)
        ]
    }

    try:
        # Gọi agent executor để xử lý input
        response = react_agent_executor.invoke(agent_input)
        print(response)
        #logger.info(f"Main Assistant Node: Phản hồi đầy đủ từ react_agent_executor: {response}")
        # Lấy câu trả lời từ response
        # Trích xuất câu trả lời cuối cùng của AI từ response
        final_ai_message_content = "Không có phản hồi từ Amber."
        if isinstance(response, dict):
            agent_messages = response.get("messages", [])
            if agent_messages and isinstance(agent_messages[-1], AIMessage):
                final_ai_message_content = agent_messages[-1].content
            else:
                logger.warning("Không tìm thấy AIMessage cuối cùng trong messages của response từ react_agent_executor.")
        else:
            logger.warning(f"Response từ react_agent_executor không phải là dict: {type(response)}")

        final_answer = final_ai_message_content

        # Fallback nếu không có câu trả lời (đã được xử lý bởi logic trên)
        if final_answer == "Không có phản hồi từ Amber." or not final_answer.strip() : # Kiểm tra kỹ hơn
            logger.warning("Final answer rỗng hoặc là fallback mặc định. Sử dụng fallback tùy chỉnh.")
            final_answer = "Ối, Amber tìm kỹ rồi mà vẫn chưa thấy thông tin bạn cần 😥. Bạn thử hỏi lại nhé!"

        logger.info(f"Main Assistant Node: Phản hồi cuối cùng đã trích xuất: '{final_answer}'")

        # Cập nhật state của graph lớn
        # messages của graph lớn sẽ là messages cũ + HumanMessage hiện tại (đã có trong state["messages"])
        # và bây giờ thêm AIMessage từ agent.
        # Cách bạn làm `state["messages"] + [AIMessage(content=final_answer)]` là ĐÚNG
        # vì state["messages"] được truyền vào node này chứa lịch sử cho đến HumanMessage hiện tại.
        
        updated_graph_messages = state.get("messages", []) + [AIMessage(content=final_answer)]

        return {
            # "messages": updated_graph_messages, # Đây là cách cập nhật messages cho graph LỚN
            # Tuy nhiên, nếu AmelaReactCompatibleAgentState được định nghĩa với MessagesPlaceholder,
            # LangGraph sẽ tự động thêm AIMessage này vào state["messages"] của graph lớn
            # nếu node trả về AIMessage trong key "messages".
            "messages": [AIMessage(content=final_answer)], # Trả về AIMessage để LangGraph tự append
            "final_answer": final_answer, # Vẫn giữ để tiện truy cập
            "clarification_needed": False
        }

    except Exception as e:
        logger.error(f"Lỗi trong Main Assistant Node: {str(e)}", exc_info=True)
        error_message = f"Xin lỗi, Amber đã gặp sự cố khi xử lý yêu cầu của bạn: {str(e)[:100]}... 😓"
        # Tương tự, cập nhật messages của graph lớn với lỗi này
        updated_graph_messages_error = state.get("messages", []) + [AIMessage(content=error_message)]
        return {
            # "messages": updated_graph_messages_error,
            "messages": [AIMessage(content=error_message)],
            "final_answer": error_message,
            "clarification_needed": False
        }
# Placeholder cho node xử lý lỗi (nếu cần)
def error_handler_node(state: AmelaReactCompatibleAgentState) -> dict: # Sửa kiểu trả về
    logger.error("--- Bắt đầu Error Handler Node ---")
    error_message = state.get("error_message", "Đã có lỗi không xác định xảy ra trong quá trình xử lý. Vui lòng thử lại.")
    logger.info(f"Error Handler Node: Thông báo lỗi: '{error_message}'")
    return {
        "messages": [AIMessage(content=error_message)], # Cập nhật messages
        "final_answer": error_message,
        "clarification_needed": False
    }

def build_graph():
    logger.info("--- Bắt đầu xây dựng Graph ---")
    workflow = StateGraph(AmelaReactCompatibleAgentState)

    logger.info("Thêm các nodes vào graph...")
    workflow.add_node("query_analyzer", query_analysis_node)
    workflow.add_node("direct_responder", direct_response_node)
    workflow.add_node("clarification_generator", clarification_node)
    workflow.add_node("main_assistant", main_assistant_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("query_analyzer")
    logger.info("Đặt entry point là 'query_analyzer'.")

    # Sửa lỗi TypeError: start_key không còn được sử dụng
    workflow.add_conditional_edges(
        "query_analyzer",           # Node nguồn
        route_after_qpa,            # Hàm điều kiện (router)
        {                           # Mapping
            "direct_response_node": "direct_responder",
            "clarification_node": "clarification_generator",
            "main_assistant_node": "main_assistant",
            "error_handler": "error_handler"
        }
    )
    logger.info("Thêm conditional edges từ 'query_analyzer' dựa trên 'route_after_qpa'.")

    workflow.add_edge("direct_responder", END)
    workflow.add_edge("clarification_generator", END)
    workflow.add_edge("main_assistant", END)
    workflow.add_edge("error_handler", END)
    logger.info("Thêm edges đến END cho 'direct_responder', 'clarification_generator', 'main_assistant', 'error_handler'.")

    # memory = SqliteSaver.from_conn_string(":memory:")
    # Lưu vào file để có thể kiểm tra session state sau này nếu cần
    checkpointer = InMemorySaver() # SỬ DỤNG IN MEMORY SAVER
    logger.info("Sử dụng InMemorySaver để test.")

    app = workflow.compile(checkpointer=checkpointer)
    # Show workflow
    #display(Image(app.get_graph().draw_mermaid_png()))
    logger.info("--- Graph đã được biên dịch thành công (với InMemorySaver) ---")
    return app

# Chạy graph
if __name__ == "__main__":
    logger.info("--- Bắt đầu chạy graph ---")
    app = build_graph()
    # Chạy graph với một state mẫu
    config={"configurable": {"thread_id": "test_thread_id"}}
    sample_state = {
        "original_query": "Tìm hiểu về quy trình quyết toán thuế TNCN tại Amela.",
        "user_roles": ["nhân viên", "quản lý"],
        "messages": [
            HumanMessage(content="Chào Amber, tôi muốn biết về quy trình quyết toán thuế TNCN."),
            AIMessage(content="Chào bạn! Tôi là Amber, trợ lý ảo Amela. Bạn cần tìm hiểu gì về quy trình này?")
        ]
    }
    result = app.invoke(sample_state, config=config)
    print(result)