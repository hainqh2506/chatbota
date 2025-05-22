
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
tool = TavilySearch(max_results=2)
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
Bạn là một chuyên gia phân tích và lập kế hoạch cho trợ lý ảo Amela.
Nhiệm vụ của bạn là đọc và phân tích câu hỏi gốc của người dùng, xác định các phần (sub-questions), lên kế hoạch trả lời, và tối ưu hóa truy vấn tìm kiếm cho từng phần.

**THÔNG TIN ĐẦU VÀO:**
- Câu hỏi gốc của người dùng: {original_query}
- Vai trò của người dùng: {user_roles}
## Quan trọng: Nếu thông tin vai trò người dùng được cung cấp, hãy sử dụng nó. Nếu không có, mặc định là ["Employee"].

**QUY TRÌNH PHÂN TÍCH & LẬP KẾ HOẠCH**
0. **Nhận diện loại câu hỏi:**
   - Nếu `original_user_query` là lời chào hỏi đơn thuần (ví dụ: "hi", "hello", "chào bạn"):
      - Đặt `intent` là "social_greeting".
      - `effective_search_query` có thể để trống hoặc chính `original_user_query`.
      - `sub_questions`, `plan_steps` có thể để trống.
      - `plan_steps` NÊN là một danh sách chứa một câu chào lại phù hợp (ví dụ: ["Chào bạn! Tôi có thể giúp gì cho bạn?"]).
      - Các trường khác như `sub_questions`, `effective_search_query`, `clarifying_questions` có thể để trống hoặc null.
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
4. **Tách `sub_questions`:** Chia câu hỏi gốc thành các phần nhỏ, mỗi phần có `text`, `intent`, `entities`, và `dependencies` (liệt kê chỉ số phụ thuộc vào các sub-question khác).
5. **Lập `plan_steps`:** Danh sách các bước cần thực hiện để trả lời từng `sub_question` theo đúng thứ tự.
6. **Xác định `clarifying_questions`:** Liệt kê những câu hỏi phụ cần hỏi lại user nếu có thông tin thiếu rõ ràng.
7. **Ước tính `complexity_level`:** Đánh giá độ phức tạp tổng thể ("low", "medium", "high").
8. **Tạo `effective_search_query`:** Truy vấn tìm kiếm tối ưu **dưới dạng danh sách**, tương ứng với từng `sub_question`.

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
  "sub_questions": [
    {{
      "text": "String",
      "intent": "String",
      "entities": ["List[String]"],
      "dependencies": ["List[Int]"]
    }}
  ],
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
  "sub_questions": [
    {{
      "text": "cách đăng ký bảo hiểm xã hội?",
      "intent": "hiểu quy trình đăng ký BHXH",
      "entities": ["đăng ký BHXH"],
      "dependencies": []
    }},
    {{
      "text": "làm thế nào để thực hiện quyết toán thuế thu nhập cá nhân?",
      "intent": "hiểu quy trình quyết toán thuế TNCN",
      "entities": ["quyết toán thuế TNCN"],
      "dependencies": [0]
    }}
  ],
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

async def query_analysis_node(state: AmelaReactCompatibleAgentState) -> AmelaReactCompatibleAgentState:
    """
    Node thực hiện phân tích câu hỏi của người dùng.
    """
    logger.info("--- Bắt đầu Query Analysis Node ---")
    original_query = state["original_query"]
    user_roles = state["user_roles"]
    # Lấy tin nhắn cuối cùng từ user để làm input cho QPA
    # Hoặc đơn giản là dùng original_query nếu đây là lượt đầu
    # Trong ADK, query được lấy từ `new_message` hoặc `initial_pipeline_state["original_user_query"]`

    logger.info(f"Phân tích câu hỏi: '{original_query}' với vai trò: {user_roles}")

    try:
        # Gọi chain để lấy kết quả phân tích có cấu trúc
        analysis_result: QueryAnalysisOutput = await query_analysis_chain.ainvoke({
            "original_query": original_query,
            "user_roles": user_roles
        })
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
            sub_questions=[],
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

async def route_after_qpa(state: AmelaReactCompatibleAgentState) -> str:
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
async def direct_response_node(state: AmelaReactCompatibleAgentState) -> AmelaReactCompatibleAgentState:
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

        if plan_steps:
            response_text = " ".join(plan_steps)
        elif intent == "social_greeting":
            response_text = "Chào bạn! Tôi là Amber trợ lý ảo Amela, rất vui được hỗ trợ bạn. 😊"
        elif intent == "chatbot_capability_query":
            response_text = "Tôi là Amber, trợ lý ảo Amela, được thiết kế để giúp bạn tìm kiếm thông tin và trả lời các câu hỏi liên quan đến nghiệp vụ của công ty mình. 💡"
        elif intent == "blocked_profanity":
            response_text = "Rất tiếc, tôi không thể xử lý yêu cầu của bạn do chứa nội dung không phù hợp. 😥"
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
async def clarification_node(state: AmelaReactCompatibleAgentState) -> AmelaReactCompatibleAgentState:
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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
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
Bạn là Amber trợ lý ảo AI thân thiện, có tổ chức và rất giỏi trong việc thực hiện kế hoạch để trả lời các câu hỏi phức tạp cho nhân viên Amela.
Bạn sẽ nhận được thông tin phân tích chi tiết từ Agent Tiền xử lý (QueryAnalysisOutput) dưới dạng một tin nhắn hệ thống hoặc tin nhắn từ user đặc biệt.
Hãy sử dụng thông tin đó, bao gồm `original_query`, `user_roles`, `asker_role_context`, `intent`, `sub_questions`, `plan_steps`, và `effective_search_query` để thực hiện.

**## THÔNG TIN PHÂN TÍCH QUERY (Từ Query Analysis Agent):**
{qpa_output_str}

**## VAI TRÒ NGƯỜI DÙNG:**
{user_roles_str} (Vai trò suy luận: {asker_role_context})

**## KẾ HOẠCH HÀNH ĐỘNG GỢI Ý (Từ Query Analysis Agent):**
{plan_steps_str}

**## QUY TRÌNH THỰC HIỆN KẾ HOẠCH (BẮT BUỘC TUÂN THỦ):**
Dựa vào thông tin phân tích ở trên, đặc biệt là `sub_questions` và `effective_search_query` tương ứng.
1.  **Xử lý Tuần tự các Câu hỏi Con (`sub_questions`)**:
    *   Với mỗi sub-question, sử dụng `effective_search_query` tương ứng để chọn tool và tìm kiếm.
    *   **Chọn Tool Phù hợp:**
        *   Ưu tiên `company_structure_tool` nếu sub-question liên quan đến cơ cấu tổ chức, phòng ban, đội nhóm, tên viết tắt.
        *   Ưu tiên `amela_documents_search_tool` cho các câu hỏi về quy trình, chính sách, kiến thức nội bộ. Nhớ rằng tool này sẽ tự động lọc theo `user_roles`.
        *   Dùng `google_search_placeholder_tool` nếu thông tin không có trong nội bộ và phù hợp tìm kiếm công khai.
    *   **Đọc kỹ kết quả từ Tool:** Tổng hợp thông tin liên quan nhất từ context để xây dựng câu trả lời mạch lạc. **Không chỉ liệt kê tên tài liệu.**
    *   **Lọc Thông tin theo Vai trò:** Dựa vào `asker_role_context`.

2.  **Tổng hợp Câu Trả lời Cuối cùng:**
    *   Kết hợp các câu trả lời cho từng sub-question thành một câu trả lời tổng thể, mạch lạc cho `original_query`.
    *   Giọng điệu: Thân thiện, tích cực, nhiệt tình. Dùng ngôi "mình", gọi người dùng là "bạn". Có thể dùng emoji 😊😉🚀💡.
    *   **Trích dẫn nguồn (BẮT BUỘC):**
        *   Tài liệu nội bộ: Ghi rõ `Source Name`. Chèn `Source URL` nếu có.
        *   Google: Chèn link Markdown.
        *   Liệt kê thành danh sách đánh số sau câu trả lời.
    *   **Xử lý khi không tìm thấy thông tin:** Nếu tool không tìm thấy gì, trả lời duyên dáng: "Ối, Amber tìm kỹ rồi mà vẫn chưa thấy thông tin bạn cần 😥..."

Hãy nhớ, bạn là Amber! Bắt đầu nào! 🚀
"""

# Langchain agent thường dùng MessagesPlaceholder. "chat_history" và "input" là keys phổ biến.
# "agent_scratchpad" được Langchain dùng để lưu các bước suy nghĩ và tool call/response.
main_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", main_assistant_prompt_str_system),
    MessagesPlaceholder(variable_name="chat_history", optional=True), # Lịch sử hội thoại
    ("human", "{input}"), # Input hiện tại, sẽ bao gồm cả thông tin QPA
    MessagesPlaceholder(variable_name="agent_scratchpad"), # Cho tool calling
])
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
# --- Tạo Langchain Agent ---
def simple_trimming_hook(state: Dict[str, Any]) -> Dict[str, Any]:
    current_messages = state.get("messages", [])
    trimmed = trim_messages(
        current_messages,
        max_tokens=1500, # Ngưỡng token, ví dụ
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True
    )
    return {"llm_input_messages": trimmed}
checkpointer = InMemorySaver()
react_agent_executor = create_react_agent(
    model=main_llm,
    tools=main_assistant_tools,
    pre_model_hook=simple_trimming_hook,
    checkpointer=checkpointer,
    verbose=True,
    state_schema=AmelaReactCompatibleAgentState,
    store=None
)

# Sử dụng create_tool_calling_agent là cách hiện đại để tạo agent có khả năng gọi tool
main_agent_runnable = create_tool_calling_agent(
    llm=main_llm,
    tools=main_assistant_tools,
    prompt=main_assistant_prompt
)

# AgentExecutor sẽ chạy agent và quản lý việc gọi tool
# `handle_parsing_errors=True` giúp agent ổn định hơn
main_agent_executor = AgentExecutor(
    agent=main_agent_runnable,
    tools=main_assistant_tools,
    verbose=True, # Để xem log chi tiết của agent
    handle_parsing_errors=True,
    max_iterations=5 # Giới hạn số lần gọi tool để tránh vòng lặp vô hạn
)


async def main_assistant_node(state: AmelaReactCompatibleAgentState) -> dict:
    """
    Node chính thực thi kế hoạch từ QPA, sử dụng tools để trả lời.
    """
    logger.info("--- Bắt đầu Main Assistant Node ---")
    query_analysis_result = state["query_analysis"]
    if not query_analysis_result:
        logger.error("Main Assistant Node: Không có Query Analysis result.")
        return {**state, "final_answer": "Lỗi: Không có thông tin phân tích để xử lý.", "clarification_needed": False}

    # Chuẩn bị input cho agent
    # Truyền toàn bộ QPA output như một phần của "input" cho agent này.
    # Hoặc có thể format nó thành một tin nhắn hệ thống/user đặc biệt.
    # Ở đây, chúng ta sẽ format nó vào prompt system và các biến khác.
    qpa_output_str = query_analysis_result.model_dump_json(indent=2)
    user_roles_str = ", ".join(query_analysis_result.user_roles or ["Employee"])
    asker_role_context = query_analysis_result.asker_role_context or "Employee"
    plan_steps_str = "\n- ".join(query_analysis_result.plan_steps or ["Không có kế hoạch cụ thể."])
    if query_analysis_result.plan_steps:
        plan_steps_str = "- " + plan_steps_str
    all_messages = state.get("messages", [])
    current_user_input_message = ""
    chat_history_for_agent = []
    if all_messages:
        if isinstance(all_messages[-1], HumanMessage):
            current_user_input_message = all_messages[-1].content
            chat_history_for_agent = all_messages[:-1]
        else: # Trường hợp đặc biệt, có thể là lỗi hoặc state khởi tạo chưa đúng
            current_user_input_message = state["original_query"] # Fallback
            chat_history_for_agent = all_messages
    try:
        # Gọi agent executor
        # AgentExecutor mong đợi input là một dict
        agent_input_dict = {
            "input": current_user_input_message, # Input cho HumanMessagePromptTemplate
            "chat_history": chat_history_for_agent, # Cho MessagesPlaceholder("chat_history")
            "qpa_output_str": qpa_output_str,
            "user_roles_str": user_roles_str,
            "asker_role_context": asker_role_context,
            "plan_steps_str": plan_steps_str,
        }

        response = await main_agent_executor.ainvoke(agent_input_dict)
        final_answer = response.get("output", "Không có phản hồi từ Amber.")

        if not final_answer: # Fallback
            final_answer = "Amber chưa thể đưa ra câu trả lời lúc này, bạn thử lại sau nhé."


        logger.info(f"Main Assistant Node: Phản hồi cuối cùng: '{final_answer}'")

        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
            "clarification_needed": False
        }

    except Exception as e:
        logger.error(f"Lỗi trong Main Assistant Node: {e}", exc_info=True)
        error_message = f"Xin lỗi, Amber đã gặp sự cố khi xử lý yêu cầu của bạn: {str(e)[:100]}..."
         
        return {
            "messages": [AIMessage(content=error_message)],
            "final_answer": error_message,
            "clarification_needed": False
        }

# Placeholder cho node xử lý lỗi (nếu cần)
async def error_handler_node(state: AmelaReactCompatibleAgentState) -> dict: # Sửa kiểu trả về
    logger.error("--- Bắt đầu Error Handler Node ---")
    error_message = state.get("error_message", "Đã có lỗi không xác định xảy ra trong quá trình xử lý. Vui lòng thử lại.")
    logger.info(f"Error Handler Node: Thông báo lỗi: '{error_message}'")
    return {
        "messages": [AIMessage(content=error_message)], # Cập nhật messages
        "final_answer": error_message,
        "clarification_needed": False
    }